import fetch from "node-fetch";

async function fetchTheAthleticJobs() {
    try {
        const response = await fetch("https://api.zensearch.jobs/api/postings", {
            "headers": {
              "accept": "*/*",
              "accept-language": "en-US,en;q=0.9",
              "authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImQ5ZTVmMmY0LTYxYjctNDY1OS1iYThmLWY4YTM0MjFiNTMyYyJ9.eyJzdWIiOiI4ZjQ2MDJlYy05MTkyLTRmYmItOWEzYi1iMWVkNmVkYzBkZmQiLCJpYXQiOjE3MzgyNTI5MjQsImV4cCI6MTczODI1NDcyNCwidXNlcl9pZCI6IjhmNDYwMmVjLTkxOTItNGZiYi05YTNiLWIxZWQ2ZWRjMGRmZCIsImlzcyI6Imh0dHBzOi8vYXV0aC56ZW5zZWFyY2guam9icyIsImVtYWlsIjoiZ3JhbW1zQHJwaS5lZHUiLCJmaXJzdF9uYW1lIjoiU2ViYXN0aWFuIiwibGFzdF9uYW1lIjoiR3JhbW1hcyIsInByb3BlcnRpZXMiOnsibWV0YWRhdGEiOnsiZGJfdXNlcl9pZCI6MzQxN319fQ.oHGt4WQTv9d5B5ITpyLl2pg4WgTs5dRQedix03yELglcPNu1b6s2bbjfCbpNRopq3kQb-VN3uSVV_HIwPaPvw5HxbghxP-y4YuUx8j0HqV3YciUz-T9k9g9VSEcf946SIoUt-M_gd7Fj2Ga2aSD_N_rsENCE6aElHFSwVfekUwxvFFu--OksnRszPIctTUW2Ji75kkLg58Gd4mVaoyWojTmN0tZram9_hWyJS2zXWNDbQPLKOCTv5h4CCXENGvfKacJdMqnA0d5YS0XeLQZXmUgAj-N2cMy4vpHXxgEi5bgLdbRKZEnpAdtTGWpNAKXc0czSiiexkNy03aB-e9Xoyg",
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
                "slug": "the-athletic-25d4a32b-6e78-40a3-b1b4-5e29f5ee3a10",
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

let jobs = await fetchTheAthleticJobs();
console.log(jobs);
