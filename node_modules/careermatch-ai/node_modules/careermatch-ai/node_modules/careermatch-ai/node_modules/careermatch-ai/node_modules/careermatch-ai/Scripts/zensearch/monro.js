import fetch from "node-fetch";

async function fetchMonroJobs() {
    try {
        const response = await fetch("https://api.zensearch.jobs/api/postings", {
            "headers": {
              "accept": "*/*",
              "accept-language": "en-US,en;q=0.9",
              "authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImQ5ZTVmMmY0LTYxYjctNDY1OS1iYThmLWY4YTM0MjFiNTMyYyJ9.eyJzdWIiOiIyMTE4ZjgzMi05MmY2LTRkMjgtYTMyNS0zZWY0MGNlNjczMGUiLCJpYXQiOjE3Mzg3MDc1NjEsImV4cCI6MTczODcwOTM2MSwidXNlcl9pZCI6IjIxMThmODMyLTkyZjYtNGQyOC1hMzI1LTNlZjQwY2U2NzMwZSIsImlzcyI6Imh0dHBzOi8vYXV0aC56ZW5zZWFyY2guam9icyIsImVtYWlsIjoiYWdhYnJpZWxjb3J1am9AZ21haWwuY29tIiwiZmlyc3RfbmFtZSI6IkFkcmlhbiIsImxhc3RfbmFtZSI6IkNvcnVqbyIsInByb3BlcnRpZXMiOnsibWV0YWRhdGEiOnsiZGJfdXNlcl9pZCI6MzQ3MX19fQ.mYxb2LneledjdZAyWZKRoReCegBnbvi_-MlB2ZuZdU5ZUKmnG4njyxjDgY2Zo4iaYACYZN6AiR6J0LvVeCmBtsrTyO-FJCfGw5U31uxaVe9IMS2CN-XmF2ccS4l8IAyVG619qjJgrZb0zN7G9MbPyYAJ-J_R6Q2i4mNV3e7oFwGHFjv2sHZVVi3YiLOp3HSn51judM_d4V-qMv3RTn-aPbA1ziWhI6S8y4SD4rvgrf-8HR8bpF8TdrxZocrlFXE62avpR9aXuPxcPExdt34YeR6GnZTz_-1-Lc0oDzqgD5MiCExVrwLlu5nvKvWQEXu77N0gW-5_0KzWU5JcDdiGhA",
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
                "slug": "monro-c9ecaa29-b33e-4753-8469-343dda89d42c",
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

let jobs = await fetchMonroJobs();
console.log(jobs);
