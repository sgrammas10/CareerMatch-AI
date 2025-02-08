import fetch from "node-fetch";
import fs from 'fs';
import { parse } from 'json2csv';
import * as cheerio from 'cheerio';
import readline from 'readline';

import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const SAVE_DIRECTORY = "C:\\Users\\...\\...\\zensearchData"; // Target directory

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

function askQuestion(query) {
    return new Promise(resolve => rl.question(query, resolve));
}

async function fetchJobs(company, authorization, slug) {
    try {
        const response = await fetch("https://api.zensearch.jobs/api/postings", {
            headers: {
                "accept": "*/*",
                "accept-language": "en-US,en;q=0.9",
                "authorization": `Bearer ${authorization}`,
                "content-type": "application/json",
                "priority": "u=1, i",
                "sec-ch-ua": "\"Not A(Brand\";v=\"8\", \"Chromium\";v=\"132\", \"Google Chrome\";v=\"132\"",
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": "\"Windows\"",
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-site",
                "Referer": "https://zensearch.jobs/",
                "Referrer-Policy": "strict-origin-when-cross-origin"
            },
            body: JSON.stringify({
                "query_type": "single_company",
                "limit": 50,
                "slug": slug,
                "since": "all",
                "skip": 0
            }),
            method: "POST"
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Failed to fetch job postings:", error);
        throw error;
    }
}

const cleanEncoding = (text) => {
    // Replace problematic characters and ensure UTF-8 compatibility
    return text
      .replace(/â€™/g, "'")   // Replace weird apostrophes
      .replace(/â€“/g, "-")   // Replace en-dash
      .replace(/[^\x00-\x7F]/g, ""); // Remove non-ASCII characters
  };
  

// Function to clean and extract text without HTML tags
const extractTextFromHtml = (htmlContent) => {
    // Remove HTML tags
    let textContent = htmlContent.replace(/<[^>]+>/g, ''); // Strip all HTML tags

    // Replace multiple consecutive spaces or line breaks with a single newline
    textContent = textContent.replace(/\s+/g, ' ').trim(); // Replace extra spaces with a single space

    // Add a newline between paragraphs based on the presence of "strong" or section breaks
    textContent = textContent.replace(/(?:\r\n|\r|\n){2,}/g, '\n\n'); // Ensure double newlines between paragraphs

    return cleanEncoding(textContent);
};

//function to save to the specified directory in csv format 
async function saveJobsToCSV(jobs, company) {
    if (!fs.existsSync(SAVE_DIRECTORY)) {
        fs.mkdirSync(SAVE_DIRECTORY, { recursive: true }); // Ensure directory exists
    }

    const jobData = jobs.postings.map(job => ({
        ID: job.id,
        Title: job.link_text,
        Link: job.link_href,
        Company: job.company?.name || 'N/A',
        Location: job.city || 'N/A',
        Remote: job.is_remote ? 'Yes' : 'No',
        Experience: job.years_of_experience || 'Not specified',
        EmploymentType: job.employment_type || 'N/A',
        DatePosted: job.created_at,
        RoleDescription: extractTextFromHtml(job.content__html)
      }));

    try {
        const csv = parse(jobData);
        const fileName = `${company}_jobs.csv`.replace(/\s+/g, '_').toLowerCase();
        const filePath = path.join(SAVE_DIRECTORY, fileName);

        await fs.promises.writeFile(filePath, csv);
        console.log(`Jobs successfully written to: ${filePath}`);
    } catch (err) {
        console.error('Error writing CSV:', err);
    }
}
 
// main 
(async () => {
    const company = await askQuestion("Enter company name: ");
    const authorization = await askQuestion("Enter authorization token: ");
    const slug = await askQuestion("Enter company slug: ");
    rl.close();

    let jobs = await fetchJobs(company, authorization, slug);
    saveJobsToCSV(jobs, company);
})();
