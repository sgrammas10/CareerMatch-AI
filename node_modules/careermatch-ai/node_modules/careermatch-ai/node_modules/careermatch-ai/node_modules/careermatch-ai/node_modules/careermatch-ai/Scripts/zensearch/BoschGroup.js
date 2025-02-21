import fetch from "node-fetch";

async function fetchBoschGroupJobs() {
    try {
        const response = await fetch("https://api.zensearch.jobs/api/postings", {
            "headers": {
              "accept": "*/*",
              "accept-language": "en-US,en;q=0.9",
              "authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImQ5ZTVmMmY0LTYxYjctNDY1OS1iYThmLWY4YTM0MjFiNTMyYyJ9.eyJzdWIiOiIyMTE4ZjgzMi05MmY2LTRkMjgtYTMyNS0zZWY0MGNlNjczMGUiLCJpYXQiOjE3Mzg3MDY3MjAsImV4cCI6MTczODcwODUyMCwidXNlcl9pZCI6IjIxMThmODMyLTkyZjYtNGQyOC1hMzI1LTNlZjQwY2U2NzMwZSIsImlzcyI6Imh0dHBzOi8vYXV0aC56ZW5zZWFyY2guam9icyIsImVtYWlsIjoiYWdhYnJpZWxjb3J1am9AZ21haWwuY29tIiwiZmlyc3RfbmFtZSI6IkFkcmlhbiIsImxhc3RfbmFtZSI6IkNvcnVqbyIsInByb3BlcnRpZXMiOnsibWV0YWRhdGEiOnsiZGJfdXNlcl9pZCI6MzQ3MX19fQ.oCMS03y8UPLFVVHIXnA80dU1Rl6J94_cu66VDRdZI1haX-92XrI4eDe84YdqKmDUg6YJpxs-FpzDYhThvTsqrqTPRdRlRZKl3lGZYeblB_RexqmxIoH11T2yhrLa7ZZyf5HZSchxHNc2u7U3sYAVZHJgfZzfUA3wSfEgpauZ75nGAJ5Ped3zG81bITHFelWtJ0x6YuuMJ0a3L-REizF1N6n8vcqAlHgteZxJMfQv562Rqgv9jxGTWpHhcN4Vso-EVMHPaqFvaVOtgkwHpAD6bWHYHFktQZ4yoPrMnyxRAI69htMAPxBHzMFYuRTHWFrpaWA7c_6RN6Q7sejwyC9jow",
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
                "slug": "bosch-group-e4fbb6e1-8823-45dd-b418-93c89aa39aeb",
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

let jobs = await fetchBoschGroupJobs();
console.log(jobs);
