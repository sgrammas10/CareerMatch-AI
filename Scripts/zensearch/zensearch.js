import fetch from "node-fetch";
import fs from "fs";
import { parse } from "json2csv";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";

// Get script directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Define paths
const DATA_DIRECTORY = path.resolve(__dirname, '../../zensearchData'); // Ensure correct path
if (!fs.existsSync(DATA_DIRECTORY)) {
    fs.mkdirSync(DATA_DIRECTORY, { recursive: true });
}

const CSV_FILE_PATH = path.join(DATA_DIRECTORY, 'company_data.csv');
const SAVE_DIRECTORY = DATA_DIRECTORY;

// Function to read company data from CSV safely
function readCompanyData() {
    if (!fs.existsSync(CSV_FILE_PATH)) {
        console.error(`❌ CSV file not found: ${CSV_FILE_PATH}`);
        process.exit(1);
    }

    const fileContent = fs.readFileSync(CSV_FILE_PATH, "utf8").trim();
    const lines = fileContent.split("\n");

    // Remove header row and parse CSV safely
    return lines.slice(1).map(line => {
        const values = line.match(/(".*?"|[^",\s]+)(?=\s*,|\s*$)/g)
            ?.map(value => value.replace(/^"|"$/g, "").trim()); // Remove quotes

        if (!values || values.length !== 3) {
            console.warn(`⚠️ Skipping malformed row: ${line}`);
            return null;
        }

        return {
            company: values[0],
            slug: values[1],
            authToken: values[2],
        };
    }).filter(entry => entry !== null);
}

// Function to fetch job postings
async function fetchJobs(company, authorization, slug) {
    try {
        console.log(`🔍 Fetching jobs for: ${company} (${slug})`);
        const response = await fetch("https://api.zensearch.jobs/api/postings", {
            method: "POST",
            headers: {
                "accept": "*/*",
                "authorization": `Bearer ${authorization}`,
                "content-type": "application/json",
                "Referer": "https://zensearch.jobs/",
            },
            body: JSON.stringify({
                "query_type": "single_company",
                "limit": 50,
                "slug": slug,
                "since": "all",
                "skip": 0,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error(`❌ Failed to fetch jobs for ${company}:`, error);
        return null; // Prevent the entire script from crashing
    }
}

// Function to save jobs to CSV
async function saveJobsToCSV(jobs, company) {
    if (!fs.existsSync(SAVE_DIRECTORY)) {
        fs.mkdirSync(SAVE_DIRECTORY, { recursive: true });
    }

    if (!jobs || !jobs.postings || jobs.postings.length === 0) {
        console.warn(`⚠️ No jobs found for ${company}. Skipping CSV creation.`);
        return;
    }

    const jobData = jobs.postings.map(job => ({
        ID: job.id,
        Title: job.link_text,
        Link: job.link_href,
        Company: job.company?.name || "N/A",
        Location: job.city || "N/A",
        Remote: job.is_remote ? "Yes" : "No",
        Experience: job.years_of_experience || "Not specified",
        EmploymentType: job.employment_type || "N/A",
        DatePosted: job.created_at,
        RoleDescription: job.content__html.replace(/<[^>]+>/g, "").trim(), // Remove HTML tags
    }));

    try {
        const csv = parse(jobData);
        const fileName = `${company}_jobs.csv`.replace(/\s+/g, "_").toLowerCase();
        const filePath = path.join(SAVE_DIRECTORY, fileName);

        await fs.promises.writeFile(filePath, csv);
        console.log(`✅ Jobs successfully saved: ${filePath}`);
    } catch (err) {
        console.error("❌ Error writing CSV:", err);
    }
}

// Main function: Process all companies in parallel
(async () => {
    const companyDataList = readCompanyData();

    if (companyDataList.length === 0) {
        console.error("❌ No valid company data found.");
        process.exit(1);
    }

    console.log(`📋 Processing ${companyDataList.length} companies...`);

    await Promise.all(
        companyDataList.map(async ({ company, slug, authToken }) => {
            const jobs = await fetchJobs(company, authToken, slug);
            if (jobs) await saveJobsToCSV(jobs, company);
        })
    );

    console.log("🎉 All companies processed!");
})();
