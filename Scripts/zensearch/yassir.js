import fetch from "node-fetch";

async function fetchYassirJobs() {
    try {
        const response = await fetch("https://api.zensearch.jobs/api/postings", {
            "headers": {
              "accept": "*/*",
              "accept-language": "en-US,en;q=0.9",
             "authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImQ5ZTVmMmY0LTYxYjctNDY1OS1iYThmLWY4YTM0MjFiNTMyYyJ9.eyJzdWIiOiI2OTgzMzI2OS1mNDViLTRlNTMtOTAwZi05OTdiZGIwNDZlNmIiLCJpYXQiOjE3Mzg3MDgyMDksImV4cCI6MTczODcxMDAwOSwidXNlcl9pZCI6IjY5ODMzMjY5LWY0NWItNGU1My05MDBmLTk5N2JkYjA0NmU2YiIsImlzcyI6Imh0dHBzOi8vYXV0aC56ZW5zZWFyY2guam9icyIsImVtYWlsIjoicGFibG9jaGFyaXphcmQ3NEBnbWFpbC5jb20iLCJmaXJzdF9uYW1lIjoiUGFibG8iLCJsYXN0X25hbWUiOiJTZW1pZGV5IiwicHJvcGVydGllcyI6eyJtZXRhZGF0YSI6eyJkYl91c2VyX2lkIjozNDcyfX19.BRB03WCNj_8GicsGpO98cqgHwnFAQD0D98sDwSsI5AZB1ONsqg0JCu0HDdIWfxr9Q0evtTKko34oAhmiVeAh2OskG0j1-q5huc1ZFxQWYV7fynKfRy_H0rmHxNEVQMPfQ4Z_ZzOOwmrCyhRnSZVJ-ntigdRZLjdIanIc0T9xVTLA6pvt7fZ6cdZ1Uc5WhfRKuvc84ElkdlFAcegGInKHylrxW6ncnuMYvHKDpxD9pztbgurQ2vkNmOGOmDd0K_7WY3P5Q4UhRvCsccpdJWtkgpdeHKliY8euZXRG_9zVBuDfI9zy7_fxLOJWdSebDCcOmfdJLA9VKlPYeQaIhgZrBg",
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
                "slug": "yassir-bfdc69b9-e25d-4eaa-9621-677849c26f1a",
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

let jobs = await fetchYassirJobs();
console.log(jobs);