import fetch from "node-fetch";

async function fetchDoordashJobs() {
    try{
        const response = await fetch("https://api.zensearch.jobs/api/postings", {
            "headers": {
              "accept": "*/*",
              "accept-language": "en-US,en;q=0.9",
              "authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImQ5ZTVmMmY0LTYxYjctNDY1OS1iYThmLWY4YTM0MjFiNTMyYyJ9.eyJzdWIiOiI4ZjQ2MDJlYy05MTkyLTRmYmItOWEzYi1iMWVkNmVkYzBkZmQiLCJpYXQiOjE3MzgyNTMwMTcsImV4cCI6MTczODI1NDgxNywidXNlcl9pZCI6IjhmNDYwMmVjLTkxOTItNGZiYi05YTNiLWIxZWQ2ZWRjMGRmZCIsImlzcyI6Imh0dHBzOi8vYXV0aC56ZW5zZWFyY2guam9icyIsImVtYWlsIjoiZ3JhbW1zQHJwaS5lZHUiLCJmaXJzdF9uYW1lIjoiU2ViYXN0aWFuIiwibGFzdF9uYW1lIjoiR3JhbW1hcyIsInByb3BlcnRpZXMiOnsibWV0YWRhdGEiOnsiZGJfdXNlcl9pZCI6MzQxN319fQ.ssE5ShVZcyu29A_W1CTBvhyJKAWxxZZlXqJiYepYXGgGy9AvmbF3dYggigyX_dfdtjdhuGIB0YDBygcpy8Qt_ccVxl0Nx9YvN7uf4477TFvlj6FtR6xQIbQ1uHXZUcnaFOBPq18HbgfddKL5Skr7VzmSE9ts2Qj9PUbj1Q3osyME9e3ubSFnzx5UziA_xykMjx3sO07kmgqsdhXcsA_jpTCwQPSUSgGarZNDQyg05F9UZf72awgCo4VV7uPyW4rse77TboCJZdyiFVkKzk9QQvE84HvH8z9PVOc_uQ3qgzOuI1ASoqxlH8gOiVhRbS40bfoxZcdySDWx0ON4RjcCgQ",
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
                "slug": "doordash-eac5f42f-5107-4be9-81b6-c0a7260fea4f",
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

let jobs = await fetchDoordashJobs();
console.log(jobs);
