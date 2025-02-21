import fetch from "node-fetch";

async function fetchAwinJobs() {
    try {
        const response = await fetch("https://api.zensearch.jobs/api/postings", {
            "headers": {
              "accept": "*/*",
              "accept-language": "en-US,en;q=0.9",
              "authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImQ5ZTVmMmY0LTYxYjctNDY1OS1iYThmLWY4YTM0MjFiNTMyYyJ9.eyJzdWIiOiIyMTE4ZjgzMi05MmY2LTRkMjgtYTMyNS0zZWY0MGNlNjczMGUiLCJpYXQiOjE3Mzg3MDQzNzgsImV4cCI6MTczODcwNjE3OCwidXNlcl9pZCI6IjIxMThmODMyLTkyZjYtNGQyOC1hMzI1LTNlZjQwY2U2NzMwZSIsImlzcyI6Imh0dHBzOi8vYXV0aC56ZW5zZWFyY2guam9icyIsImVtYWlsIjoiYWdhYnJpZWxjb3J1am9AZ21haWwuY29tIiwiZmlyc3RfbmFtZSI6IkFkcmlhbiIsImxhc3RfbmFtZSI6IkNvcnVqbyIsInByb3BlcnRpZXMiOnsibWV0YWRhdGEiOnsiZGJfdXNlcl9pZCI6MzQ3MX19fQ.hPHibXZFDFoU7FJSi2Sxty6yvRS8J8Me3x4D1KuNyBwcxT7Eg0QK0rnbplhrYxRBmT4HzW70JosMwDYbrxxAtOta7p9t2-UcEsrFs0Bue-ZyBbTHmgNq7k7E4AA0HhUARdNU2ttt0PpHrMvdGdxTFwgPlyXtGN4flELTUM5Ub6xzJSXd9TiWzAD3_ZR0R8r3g-i8LtlypjePEUGDvEyxxgr9oqTIRWKJtRw5aYTkZwUMy-Ap7mNW5xJGp8TmsN_myn_4GmZEr6CNH9HkQ4Lx2lFst16wYg9CGMQ3PoK68XbS5V0r6m2lc9e22kUFG-DmWR2Vw8PXM18vYdTyY_hUFA",
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
                "slug": "awin-dda1368c-0785-4e52-b42b-7c56dec7987d",
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
        console.error("Failed to fetch Awin job postings:", error);
        throw error;
    }
}

let jobs = await fetchAwinJobs();
console.log(jobs);
