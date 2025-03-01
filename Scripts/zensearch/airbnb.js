/*
    Largely serves as the template for scraping job data from a website
    This includes
        - Job title
        - Location
        - Pay range
        - All relevant and non relevant job description data

*/



import fetch from "node-fetch";
import fs from 'fs';
import { parse } from 'json2csv';
import * as cheerio from 'cheerio';

import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);


async function fetchAirbnbJobs() {
    try {
        const response = await fetch("https://api.zensearch.jobs/api/postings", {
            headers: {
                "accept": "*/*",
                "accept-language": "en-US,en;q=0.9",
                "authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImQ5ZTVmMmY0LTYxYjctNDY1OS1iYThmLWY4YTM0MjFiNTMyYyJ9.eyJzdWIiOiI4ZjQ2MDJlYy05MTkyLTRmYmItOWEzYi1iMWVkNmVkYzBkZmQiLCJpYXQiOjE3MzgyNTI2NjQsImV4cCI6MTczODI1NDQ2NCwidXNlcl9pZCI6IjhmNDYwMmVjLTkxOTItNGZiYi05YTNiLWIxZWQ2ZWRjMGRmZCIsImlzcyI6Imh0dHBzOi8vYXV0aC56ZW5zZWFyY2guam9icyIsImVtYWlsIjoiZ3JhbW1zQHJwaS5lZHUiLCJmaXJzdF9uYW1lIjoiU2ViYXN0aWFuIiwibGFzdF9uYW1lIjoiR3JhbW1hcyIsInByb3BlcnRpZXMiOnsibWV0YWRhdGEiOnsiZGJfdXNlcl9pZCI6MzQxN319fQ.BAXefymu-lT2L_fJGri9MjWtx81pvB1M76BZ9LUjgAR0OZyYtDKaNx2-U-AKP9fqxVXKSo7CQN_9xrPL0w-jIoLiU4DYiRAUfQOMG9ReZhaSplWKj2NtlGtsU_jtNZcfdsj5EdCR6mA5jewdnp0jFIIe08IJR_-JfysyiA_CkbH3xl6xPLT6RqTWXh4pbaQwY07PF62mzRkejch2oLhq-XIWbSh9cQD4JQ07UdmJkJh4SfOGgH_UhTyPU-lu9-Q6xsSDP1VKY0ozy5r4Z7ims6Ggxe267UOw0isCZj5lQULDBEP3WDFnE13aSGjmWDYcTyQsdalIgPyywCQDrwi_XA",
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
                "slug": "airbnb-3f8ef1ed-15de-4931-9940-5707a0f4da66",
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
        console.error("Failed to fetch Airbnb job postings:", error);
        throw error;
    }
}

function cleanText(text) {
    if (typeof text !== 'string') {
        text = String(text);  // Cast non-string input to a string
    }

    return text
        .normalize("NFKD") // Normalize Unicode characters
        .replace(/â€™/g, "'") // Replace special apostrophe
        .replace(/â€“/g, "-") // Replace dashes
        .replace(/[^a-zA-Z0-9.,!?()'"%-\s]/g, '') // Remove unwanted symbols
        .trim();
}
function parseHtmlContent(htmlContent) {
    const $ = cheerio.load(htmlContent);
    let sections = {};

    // Loop through each paragraph or list
    $("p, ul").each((index, element) => {
        let text = cleanText($(element).text().trim());
        if (!text) return; // Skip empty elements

        // Check if the text is a label that needs merging with content
        if (/^(Your Expertise:|Your Location:|Job Type:|Required Skills:)/i.test(text)) {
            // Treat as label and add to sections
            sections[text] = cleanText($(element).next().text().trim()) || "N/A";
        } else {
            // Generic section - add it as Section_{index}
            sections[`Section_${index}`] = text;
        }
    });

    return sections;
}

async function saveJobsToCSV(jobs) {
    const jobData = jobs.postings.map(job => {
        let parsedContent = parseHtmlContent(job.content__html || {});

        // Combine all parsed sections into a single text block
        const jobDescription = Object.entries(parsedContent)
            .map(([key, value]) => `${key}: ${value}`)
            .join("\n"); // Joins everything into one text block

        return {
            ID: job.id,
            Title: cleanText(job.link_text),
            Link: job.link_href,
            Company: cleanText(job.company?.name || 'N/A'),
            Location: cleanText(job.city || 'N/A'),
            Remote: job.is_remote ? 'Yes' : 'No',
            Experience: cleanText(job.years_of_experience || 'Not specified'),
            EmploymentType: cleanText(job.employment_type || 'N/A'),
            DatePosted: job.created_at,
            Description: jobDescription // Single column for all job details
        };
    });

    try {
        const directoryPath = path.resolve(__dirname, '../../zensearchData'); // Ensure correct path
        if (!fs.existsSync(directoryPath)) {
            fs.mkdirSync(directoryPath, { recursive: true });
        }

        const filePath = path.join(directoryPath, 'airbnb_jobs.csv');
        const csv = parse(jobData);
        fs.writeFileSync(filePath, csv);
        console.log(`Jobs successfully written to ${filePath}`);
    } catch (err) {
        console.error('Error writing CSV:', err);
    }
}



(async () => {
    let jobs = await fetchAirbnbJobs();
    saveJobsToCSV(jobs);
})();
