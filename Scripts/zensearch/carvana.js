import fetch from "node-fetch";

async function fetchCarvanaJobs() {
    try {
        const response = await fetch("https://api.zensearch.jobs/api/postings", {
            "headers": {
              "accept": "*/*",
              "accept-language": "en-US,en;q=0.9",
              "authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImQ5ZTVmMmY0LTYxYjctNDY1OS1iYThmLWY4YTM0MjFiNTMyYyJ9.eyJzdWIiOiI2OTgzMzI2OS1mNDViLTRlNTMtOTAwZi05OTdiZGIwNDZlNmIiLCJpYXQiOjE3Mzg3MDc5MTcsImV4cCI6MTczODcwOTcxNywidXNlcl9pZCI6IjY5ODMzMjY5LWY0NWItNGU1My05MDBmLTk5N2JkYjA0NmU2YiIsImlzcyI6Imh0dHBzOi8vYXV0aC56ZW5zZWFyY2guam9icyIsImVtYWlsIjoicGFibG9jaGFyaXphcmQ3NEBnbWFpbC5jb20iLCJmaXJzdF9uYW1lIjoiUGFibG8iLCJsYXN0X25hbWUiOiJTZW1pZGV5IiwicHJvcGVydGllcyI6eyJtZXRhZGF0YSI6eyJkYl91c2VyX2lkIjozNDcyfX19.Af6gtWVjMBHbeBoxfzwjcXzq841Kg71PeznWguYPxJXjc1HqF_ww1EbtEsBffOTf3lEDl4wxUZWAHIUfSMiI4mog_N1x6l-1_FDXQ8CvuYKy3L1gVbByYqoajuo2v0gj2vnS60yqIARhRCVRTjTqseS7n7gVUhPVYDaGdGC8C31xyP3FoH95zN587tc-ZQKWXHdm-jpBTWGd0hAYpY4pK4il4a_JLLQEMKfiZwAUMBnNxR6wg79MYM1SDC1RKqHVFuw8up63mVUM7Y2_ZCZX8QqLnkKsrWE6f03OZH2ySWKIaPVfyA2ockQJUuJxIGw8XUDiyLK3-muERG0nzZuzZA",
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
                "slug": "carvana-b092b318-7437-43f1-b258-bea14a10e152",
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

let jobs = await fetchCarvanaJobs();
console.log(jobs);