import fetch from "node-fetch";
import fs from 'fs';
import { parse } from 'json2csv';

// Function to clean up encoding issues by replacing non-UTF8 characters
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

async function fetchRedditJobs(){
    try{
        const response = await fetch("https://api.zensearch.jobs/api/postings", {
            "headers": {
              "accept": "*/*",
              "accept-language": "en-US,en;q=0.9",
              "authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImQ5ZTVmMmY0LTYxYjctNDY1OS1iYThmLWY4YTM0MjFiNTMyYyJ9.eyJzdWIiOiI4ZjQ2MDJlYy05MTkyLTRmYmItOWEzYi1iMWVkNmVkYzBkZmQiLCJpYXQiOjE3MzgyNTI1NjcsImV4cCI6MTczODI1NDM2NywidXNlcl9pZCI6IjhmNDYwMmVjLTkxOTItNGZiYi05YTNiLWIxZWQ2ZWRjMGRmZCIsImlzcyI6Imh0dHBzOi8vYXV0aC56ZW5zZWFyY2guam9icyIsImVtYWlsIjoiZ3JhbW1zQHJwaS5lZHUiLCJmaXJzdF9uYW1lIjoiU2ViYXN0aWFuIiwibGFzdF9uYW1lIjoiR3JhbW1hcyIsInByb3BlcnRpZXMiOnsibWV0YWRhdGEiOnsiZGJfdXNlcl9pZCI6MzQxN319fQ.AUnfDO6Acyz8F-fR7dVXBvfVVk-R2ohGR04-rO70INzsfvI7fGW_elZQfo0yPzkBk8yOM9IJhH5kyUutaLFOR0yUcNcUJuIKTW1Ud61EUylF3UU4ubHLByQEsnFxL0jHb78YbGstOOmy5_qr9ayipT9cDUcOVrgIUNfQvgFe2_fhqD7dl0J033PXsrZLkcXxH8MvPiLzrluEk6NkP3gzX1sLvgGgAvv_xvyceXRjbpGgSNKxwEhLDMpwyW6JlQckCrFM48AIMRqm0HRRCte3vr5CSAUHVPTAGMSZmW1j7fuJ6eoolvn0ImbOxHyhNYuzw3qVggvoxLZTu3yXcTgjPw",
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
            "body": JSON.stringify({
                "query_type": "single_company",
                "limit": 50,
                "slug": "reddit-dad09ab1-e759-4b0a-8332-00ddf322ac81",
                "since": "all",
                "skip": 0
            }),
            "method": "POST"
          });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Failed to fetch Peloton job postings:", error);
        throw error;
    }

}

async function saveJobsToCSV(jobs){

    // Map the jobs to extract relevant fields
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
    // Convert JSON to CSV
    const csv = parse(jobData);
    
    // Write CSV to a file
    fs.writeFile('reddit_jobs.csv', csv, (err) => {
      if (err) {
        console.error('Error writing to file:', err);
      } else {
        console.log('Jobs successfully written to reddit_jobs.csv');
      }
    });
  } catch (err) {
    console.error('Error parsing JSON to CSV:', err);
  }
}

let jobs = await fetchRedditJobs();
console.log(jobs);

saveJobsToCSV(jobs)