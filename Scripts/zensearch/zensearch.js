import fetch from "node-fetch";
import fs from "fs";
import { parse } from "json2csv";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";
import translate from "google-translate-api-x";

// Get script directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Define paths
const DATA_DIRECTORY = path.resolve(__dirname, '../../zensearchData'); // Ensure correct path
if (!fs.existsSync(DATA_DIRECTORY)) {
    fs.mkdirSync(DATA_DIRECTORY, { recursive: true });
}

const INDIVIDUAL_CSV_DIRECTORY = path.join(DATA_DIRECTORY, 'individual_csvs');
// Create the individual_csvs directory if it doesn't exist
if (!fs.existsSync(INDIVIDUAL_CSV_DIRECTORY)) {
    fs.mkdirSync(INDIVIDUAL_CSV_DIRECTORY, { recursive: true });
}

const CSV_FILE_PATH = path.join(DATA_DIRECTORY, 'company_data.csv');
const ALL_JOBS_CSV_PATH = path.join(DATA_DIRECTORY, 'job_postings.csv');

async function translateToEnglish(text, retries = 3) {
    let chunks = [];
    let translatedChunks = [];
    let start = 0;
    let maxLength = 5000;

    // Helper function to delay execution
    const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    // Split text into manageable chunks
    while (start < text.length) {
        let end = start + maxLength;

        if (end < text.length) {
            // Find the last sentence-ending punctuation within the limit
            let lastPunctuation = Math.max(
                text.lastIndexOf(".", end),
                text.lastIndexOf("!", end),
                text.lastIndexOf("?", end)
            );

            if (lastPunctuation > start) {
                end = lastPunctuation + 1; // Include punctuation
            } else {
                let lastSpace = text.lastIndexOf(" ", end);
                if (lastSpace > start) {
                    end = lastSpace;
                }
            }
        }

        chunks.push(text.substring(start, end).trim());
        start = end + 1;
    }

    // Translate each chunk with rate limiting
    for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        let translatedText = "";
        
        for (let attempt = 0; attempt < retries; attempt++) {
            try {
                // Add a progressively longer delay between requests
                if (attempt > 0) {
                    // Exponential backoff: 2^attempt * 1000 ms (1s, 2s, 4s, etc.)
                    const backoffDelay = Math.pow(2, attempt) * 1000;
                    await delay(backoffDelay);
                    console.log(`Retrying after ${backoffDelay}ms delay...`);
                }
                
                const result = await translate(chunk, { to: "en" });
                translatedText = result.text;
                break; // Exit retry loop on success
            } catch (error) {
                console.error(`Translation error (attempt ${attempt + 1}):`, error);
                
                if (error.message && (
                    error.message.includes("too many requests") ||
                    error.message.includes("rate limit") ||
                    error.message.includes("429")
                )) {
                    // If rate limited, always wait before retrying
                    const rateLimit = (attempt + 1) * 2000; // Increasing delay: 2s, 4s, 6s
                    console.log(`Rate limit detected, waiting ${rateLimit}ms before retry...`);
                    await delay(rateLimit);
                }
                
                if (attempt === retries - 1) {
                    translatedText = "[Translation failed]";
                }
            }
        }
        
        translatedChunks.push(translatedText);
        
        // Add delay between chunk translations (500ms)
        if (i < chunks.length - 1) {
            await delay(500);
        }
    }

    return translatedChunks.join(" "); // Return the full translated text
}

// Function to read company data from CSV safely
function readCompanyData() {
    if (!fs.existsSync(CSV_FILE_PATH)) {
        console.error(`CSV file not found: ${CSV_FILE_PATH}`);
        process.exit(1);
    }

    const fileContent = fs.readFileSync(CSV_FILE_PATH, "utf8").trim();
    const lines = fileContent.split("\n");

    // Remove header row and parse CSV safely
    return lines.slice(1).map(line => {
        const values = line.match(/(".*?"|[^",\s]+)(?=\s*,|\s*$)/g)
            ?.map(value => value.replace(/^"|"$/g, "").trim()); // Remove quotes

        if (!values || values.length !== 3) {
            console.warn(`Skipping malformed row: ${line}`);
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
        console.log(`Fetching jobs for: ${company} (${slug})`);
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
        console.error(`Failed to fetch jobs for ${company}:`, error);
        return null; // Prevent the entire script from crashing
    }
}

// Function to process job data
async function processJobData(jobs, company) {
    if (!jobs || !jobs.postings || jobs.postings.length === 0) {
        console.warn(`No jobs found for ${company}. Skipping.`);
        return [];
    }

    return await Promise.all(jobs.postings.map(async job => ({
        ID: job.id,
        Title: await job.link_text,
        Link: job.link_href,
        Company: company || job.company?.name || "N/A",
        Location: job.city || "N/A",
        Remote: job.is_remote ? "Yes" : "No",
        Experience: job.years_of_experience || "Not specified",
        EmploymentType: job.employment_type || "N/A",
        DatePosted: job.created_at,
        RoleDescription: await 
            job.content__html
                .replace(/<[^>]+>/g, "") // Remove HTML
                .normalize("NFKC") // Normalize Unicode
                .trim()
                .replace(/[^\x00-\x7F]/g, "") // Remove extra characters
        ,
    })));
}

// Function to save jobs to company-specific CSV
async function saveJobsToCSV(jobData, company) {
    if (!jobData || jobData.length === 0) {
        console.warn(`No job data to save for ${company}.`);
        return;
    }

    try {
        const csv = parse(jobData);
        const fileName = `${company}_jobs.csv`.replace(/\s+/g, "_").toLowerCase();
        const filePath = path.join(INDIVIDUAL_CSV_DIRECTORY, fileName);

        await fs.promises.writeFile(filePath, csv);
        console.log(`Jobs successfully saved: ${filePath}`);
    } catch (err) {
        console.error(`Error writing CSV for ${company}:`, err);
    }
}

// Function to save all jobs to a single CSV
async function saveAllJobsToCSV(allJobData) {
    if (!allJobData || allJobData.length === 0) {
        console.error("No job data to save to combined CSV.");
        return;
    }

    try {
        const csv = parse(allJobData);
        await fs.promises.writeFile(ALL_JOBS_CSV_PATH, csv);
        console.log(`All jobs successfully saved to: ${ALL_JOBS_CSV_PATH}`);
    } catch (err) {
        console.error("Error writing combined CSV:", err);
    }
}

// Main function: Process all companies and collect all job data
(async () => {
    const companyDataList = readCompanyData();
    let allJobData = [];

    if (companyDataList.length === 0) {
        console.error("No valid company data found.");
        process.exit(1);
    }

    console.log(`Processing ${companyDataList.length} companies...`);

    // Process each company sequentially to better manage API rate limits
    for (const { company, slug, authToken } of companyDataList) {
        const jobs = await fetchJobs(company, authToken, slug);
        if (jobs) {
            const jobData = await processJobData(jobs, company);
            
            // Save company-specific CSV to individual_csvs directory
            await saveJobsToCSV(jobData, company);
            
            // Add to combined job data
            allJobData = allJobData.concat(jobData);
        }
    }

    // Save all jobs to combined CSV
    await saveAllJobsToCSV(allJobData);

    console.log("All companies processed!");
})();