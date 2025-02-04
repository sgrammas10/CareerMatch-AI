import fetch from "node-fetch";

async function fetchAccorJobs() {
    try {
        const response = await fetch("https://api.zensearch.jobs/api/postings", {
            "headers": {
              "accept": "*/*",
              "accept-language": "en-US,en;q=0.9",
              "authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImQ5ZTVmMmY0LTYxYjctNDY1OS1iYThmLWY4YTM0MjFiNTMyYyJ9.eyJzdWIiOiI2OTgzMzI2OS1mNDViLTRlNTMtOTAwZi05OTdiZGIwNDZlNmIiLCJpYXQiOjE3Mzg3MDY3MTAsImV4cCI6MTczODcwODUxMCwidXNlcl9pZCI6IjY5ODMzMjY5LWY0NWItNGU1My05MDBmLTk5N2JkYjA0NmU2YiIsImlzcyI6Imh0dHBzOi8vYXV0aC56ZW5zZWFyY2guam9icyIsImVtYWlsIjoicGFibG9jaGFyaXphcmQ3NEBnbWFpbC5jb20iLCJmaXJzdF9uYW1lIjoiUGFibG8iLCJsYXN0X25hbWUiOiJTZW1pZGV5IiwicHJvcGVydGllcyI6eyJtZXRhZGF0YSI6eyJkYl91c2VyX2lkIjozNDcyfX19.lX9F__spQdQAwWc_MDAaHFHkZ8N45qzSww1ixauEPenyltqK9Vk6BCVQEVEJ5j_viDD5C8QYJ0WgeYK-nPQ5WdDBiYs8fSF2OOqDq4vLtnhkHaVPalYcyvR62XuZw6MeyOq_FiNjszWynurnQEdAnvSverWcgMRnPFm5HytkeBoyHhUJnmtI50CoicmjNxUQwcndfA5Ku3gmGrxoDYyiaZ-HTVMrhBzw7Q-t3uL9_alRWl4PWFElRWzfvFZ4KZM3DIEhUWbietk7Rzlwi0IGx5wFlx2vRhUqLx8cK47WsSFTxQOIhEv31VPLmh6QHY9A9MsFhFRjokVKZXak06Vptw",
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
                "slug": "accorcorpo-f3ee215c-2317-41f2-b11f-808c02435177",
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

let jobs = await fetchAccorJobs();
console.log(jobs);