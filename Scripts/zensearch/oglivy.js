import fetch from "node-fetch";

async function fetchOglivyJobs() {
    try {
        const response = await fetch("https://api.zensearch.jobs/api/postings", {
            "headers": {
              "accept": "*/*",
              "accept-language": "en-US,en;q=0.9",
              "authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImQ5ZTVmMmY0LTYxYjctNDY1OS1iYThmLWY4YTM0MjFiNTMyYyJ9.eyJzdWIiOiI2OTgzMzI2OS1mNDViLTRlNTMtOTAwZi05OTdiZGIwNDZlNmIiLCJpYXQiOjE3Mzg3MDg0OTQsImV4cCI6MTczODcxMDI5NCwidXNlcl9pZCI6IjY5ODMzMjY5LWY0NWItNGU1My05MDBmLTk5N2JkYjA0NmU2YiIsImlzcyI6Imh0dHBzOi8vYXV0aC56ZW5zZWFyY2guam9icyIsImVtYWlsIjoicGFibG9jaGFyaXphcmQ3NEBnbWFpbC5jb20iLCJmaXJzdF9uYW1lIjoiUGFibG8iLCJsYXN0X25hbWUiOiJTZW1pZGV5IiwicHJvcGVydGllcyI6eyJtZXRhZGF0YSI6eyJkYl91c2VyX2lkIjozNDcyfX19.H2CN-N4qp-FPpa9IrdRvrLgrhFj81ZouDH-M-vkYGlJu4LHuwwWcb60rpS2Kyu5szcLln0czdDuyv4oDZzsZXHn9yoego1AbrVlqJ7qboVRl_Zepl1N6vkeHIj6Hni-9UEahPJIyEhoJu0tvstf9feNGA4Loj-aYGximc9W8wi-celZ4aAYy01zeKWylPj-GwXKxnOYfT6YXGrdLzr6OduapqLuIq1w5_sC0I2SbWo69Q89fJ20bnqXXT-QWYVSATVJxaLhMCsvaGfQv4oJCPibMGXqzVp06SRONDzC1WonQKWSd22LZsmlkty-sCyQLNJ76tbnbz0p6XYQ0zhZvyA",
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
                "slug": "oglivy-48993512-7058-494b-8d19-ee87215fa5e9",
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

let jobs = await fetchOglivyJobs();
console.log(jobs);