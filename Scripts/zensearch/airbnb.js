import fetch from "node-fetch";

async function fetchAirbnbJobs() {
    try {
        const response = await fetch("https://api.zensearch.jobs/api/postings", {
            "headers": {
              "accept": "*/*",
              "accept-language": "en-US,en;q=0.9",
              "authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImQ5ZTVmMmY0LTYxYjctNDY1OS1iYThmLWY4YTM0MjFiNTMyYyJ9.eyJzdWIiOiI4ZjQ2MDJlYy05MTkyLTRmYmItOWEzYi1iMWVkNmVkYzBkZmQiLCJpYXQiOjE3MzgyNTI2NjQsImV4cCI6MTczODI1NDQ2NCwidXNlcl9pZCI6IjhmNDYwMmVjLTkxOTItNGZiYi05YTNiLWIxZWQ2ZWRjMGRmZCIsImlzcyI6Imh0dHBzOi8vYXV0aC56ZW5zZWFyY2guam9icyIsImVtYWlsIjoiZ3JhbW1zQHJwaS5lZHUiLCJmaXJzdF9uYW1lIjoiU2ViYXN0aWFuIiwibGFzdF9uYW1lIjoiR3JhbW1hcyIsInByb3BlcnRpZXMiOnsibWV0YWRhdGEiOnsiZGJfdXNlcl9pZCI6MzQxN319fQ.BAXefymu-lT2L_fJGri9MjWtx81pvB1M76BZ9LUjgAR0OZyYtDKaNx2-U-AKP9fqxVXKSo7CQN_9xrPL0w-jIoLiU4DYiRAUfQOMG9ReZhaSplWKj2NtlGtsU_jtNZcfdsj5EdCR6mA5jewdnp0jFIIe08IJR_-JfysyiA_CkbH3xl6xPLT6RqTWXh4pbaQwY07PF62mzRkejch2oLhq-XIWbSh9cQD4JQ07UdmJkJh4SfOGgH_UhTyPU-lu9-Q6xsSDP1VKY0ozy5r4Z7ims6Ggxe267UOw0isCZj5lQULDBEP3WDFnE13aSGjmWDYcTyQsdalIgPyywCQDrwi_XA",
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
                "slug": "airbnb-3f8ef1ed-15de-4931-9940-5707a0f4da66",
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

let jobs = await fetchAirbnbJobs();
console.log(jobs);
