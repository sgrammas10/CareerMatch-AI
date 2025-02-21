import fetch from "node-fetch";

async function fetchCryptoComJobs() {
    try{
        const response = await fetch("https://api.zensearch.jobs/api/postings", {
            "headers": {
                "accept": "*/*",
                "accept-language": "en-US,en;q=0.9",
                "authorization": "Bearer YOUR_BEARER_TOKEN",
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
                "slug": "cryptocom-b5037459-0157-4033-9893-3a1f1aec722f",
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

let jobs = await fetchCryptoComJobs();
console.log(jobs)