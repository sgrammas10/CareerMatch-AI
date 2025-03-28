import fetch from "node-fetch";

async function fetchSpaceXJobs() {
    try {
        const response = await fetch("https://api.zensearch.jobs/api/postings", {
            "headers": {
                "accept": "*/*",
                "accept-language": "en-US,en;q=0.9",
                "authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImQ5ZTVmMmY0LTYxYjctNDY1OS1iYThmLWY4YTM0MjFiNTMyYyJ9.eyJzdWIiOiI4ZjQ2MDJlYy05MTkyLTRmYmItOWEzYi1iMWVkNmVkYzBkZmQiLCJpYXQiOjE3MzgyNTE4MTYsImV4cCI6MTczODI1MzYxNiwidXNlcl9pZCI6IjhmNDYwMmVjLTkxOTItNGZiYi05YTNiLWIxZWQ2ZWRjMGRmZCIsImlzcyI6Imh0dHBzOi8vYXV0aC56ZW5zZWFyY2guam9icyIsImVtYWlsIjoiZ3JhbW1zQHJwaS5lZHUiLCJmaXJzdF9uYW1lIjoiU2ViYXN0aWFuIiwibGFzdF9uYW1lIjoiR3JhbW1hcyIsInByb3BlcnRpZXMiOnsibWV0YWRhdGEiOnsiZGJfdXNlcl9pZCI6MzQxN319fQ.HLoI6Tg5-qDMDU3kVMH6j2WkHAdVFj5QsBTWuzIqd6RY7EdqK9TC3tQyIkT1LpLiuLc-6Y5ScBpSaO1lixQvZIfknaku73rGXVPLPyIvjsNClGwrRwJAc7VYQV0yCdL36mZz5rvuVGdCuOMDGWVeqA3tvmeYfjw-_2RxZ9H4LkgtC3mRLg2H1RUtnX0ML-XWhl0-OwkgGZdev1u6nNqZ9Mj3b-SNZcd7frhoMh1erZobZj6cUHNG5JFyExd-HSKT-8YQGaQbxFi418mj4p23D2YOrNkso5qL7V_cVKIsLJsY2Qq8nwBwyg15TN6I2yC8kCNULokjABF4PukiwtbOYA",
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
                "slug": "spacex-91f5acb0-e680-499a-807b-54b1b74722e3",
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
        console.error("Failed to fetch SpaceX job postings:", error.message);
        throw error;
    }
}

let jobs = await fetchSpaceXJobs();
console.log(jobs);
 