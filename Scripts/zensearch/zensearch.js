import fetch from "node-fetch";
import fs from 'fs';
import { parse } from 'json2csv';
import * as cheerio from 'cheerio';
import readline from 'readline';

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

const cleanText = (text) => {
    if (typeof text !== "string") return "";

    return text
        .replace(/â€™/g, "'")   // Replace weird apostrophes
        .replace(/â€“/g, "-")   // Replace en-dash
        .replace(/[^\x00-\x7F]/g, ""); // Remove non-ASCII characters
};

const extractTextFromHtml = (htmlContent) => {
    if (typeof htmlContent !== "string") return "";

    // Remove HTML tags
    let textContent = htmlContent.replace(/<[^>]+>/g, ''); // Strip all HTML tags

    // Replace multiple consecutive spaces or line breaks with a single space
    textContent = textContent.replace(/\s+/g, ' ').trim();

    // Add a newline between paragraphs for section breaks
    textContent = textContent.replace(/(?:\r\n|\r|\n){2,}/g, '\n\n'); 

    return cleanText(textContent);
};

const parseHtmlContent = (htmlContent) => {
    if (typeof htmlContent !== "string") return {};

    const $ = cheerio.load(htmlContent);
    let sections = {};

    $("p, ul").each((index, element) => {
        let text = extractTextFromHtml($(element).text().trim());
        if (!text) return;

        if (/^(Your Expertise:|Your Location:|Job Type:|Required Skills:)/i.test(text)) {
            sections[text] = extractTextFromHtml($(element).next().text().trim()) || "N/A";
        } else {
            sections[`Section_${index}`] = text;
        }
    });

    return sections;
};

async function saveJobsToCSV(jobs, company) {
    const jobData = jobs.postings.map(job => {
        let parsedContent = parseHtmlContent(job.content__html || {});
        const sortedSections = {};
        Object.keys(parsedContent).sort().forEach((key) => {
            sortedSections[key] = parsedContent[key];
        });

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
            ...sortedSections
        };
    });

    try {
        const csv = parse(jobData);
        const fileName = `${company}_jobs.csv`.replace(/\s+/g, '_').toLowerCase();
        fs.writeFileSync(fileName, csv);
        console.log(`Jobs successfully written to ${fileName}`);
    } catch (err) {
        console.error('Error writing CSV:', err);
    }
}

(async () => {
    const company = await askQuestion("Enter company name: ");
    const authorization = await askQuestion("Enter authorization token: ");
    const slug = await askQuestion("Enter company slug: ");
    rl.close();

    let jobs = await fetchJobs(company, authorization, slug);
    saveJobsToCSV(jobs, company);
})();