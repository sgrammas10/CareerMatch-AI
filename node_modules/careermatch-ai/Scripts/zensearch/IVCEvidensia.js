import fetch from "node-fetch";

async function fetchIVCEvidensiaJobs() {
    try {
        const response = await fetch("https://api.zensearch.jobs/api/postings", {
            "headers": {
              "accept": "*/*",
              "accept-language": "en-US,en;q=0.9",
              "authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImQ5ZTVmMmY0LTYxYjctNDY1OS1iYThmLWY4YTM0MjFiNTMyYyJ9.eyJzdWIiOiIyMTE4ZjgzMi05MmY2LTRkMjgtYTMyNS0zZWY0MGNlNjczMGUiLCJpYXQiOjE3Mzg3MDczMzksImV4cCI6MTczODcwOTEzOSwidXNlcl9pZCI6IjIxMThmODMyLTkyZjYtNGQyOC1hMzI1LTNlZjQwY2U2NzMwZSIsImlzcyI6Imh0dHBzOi8vYXV0aC56ZW5zZWFyY2guam9icyIsImVtYWlsIjoiYWdhYnJpZWxjb3J1am9AZ21haWwuY29tIiwiZmlyc3RfbmFtZSI6IkFkcmlhbiIsImxhc3RfbmFtZSI6IkNvcnVqbyIsInByb3BlcnRpZXMiOnsibWV0YWRhdGEiOnsiZGJfdXNlcl9pZCI6MzQ3MX19fQ.hNb1UIoA1S0mZvx6dtG6SdyCfIwFcPHPQIQaYa59F9nVfrYJDf4zEnC8zdNApd5gUsUv_KCWOHvnuaMSzVh0yPcI32PdOMygu7PXnVolcmU8f1_uqoxuFFTZRI00G2y350v-6-LBzN6diE1vXtes3Bf_ilsmlE1Yfz6FpnWaGW2of1VRpTgbPVSNU_M-eBJDI5VAnA4gIZHtm1QJ-BGeZQwupquNHr_r2aXjgqAs12DiVlBJkQLhrh331syuUp9lIF3vyZ8LWVxgo6yMd_yEwYQwMk9YZPikmxggjZDB-mSpnQ9hCJztDEo-YZIp06RuvVf0UMILas9RNoALQ0ukjw",
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
                "slug": "ivc-evidensia-2608dc7d-27b4-4b3a-9f70-84f8387c9c93",
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

let jobs = await fetchIVCEvidensiaJobs();
console.log(jobs);
