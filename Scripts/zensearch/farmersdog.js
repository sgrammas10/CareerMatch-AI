fetch("https://api.zensearch.jobs/api/postings", {
    "headers": {
      "accept": "*/*",
      "accept-language": "en-US,en;q=0.9",
      "authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImQ5ZTVmMmY0LTYxYjctNDY1OS1iYThmLWY4YTM0MjFiNTMyYyJ9.eyJzdWIiOiI4ZjQ2MDJlYy05MTkyLTRmYmItOWEzYi1iMWVkNmVkYzBkZmQiLCJpYXQiOjE3MzgyNTIzMzksImV4cCI6MTczODI1NDEzOSwidXNlcl9pZCI6IjhmNDYwMmVjLTkxOTItNGZiYi05YTNiLWIxZWQ2ZWRjMGRmZCIsImlzcyI6Imh0dHBzOi8vYXV0aC56ZW5zZWFyY2guam9icyIsImVtYWlsIjoiZ3JhbW1zQHJwaS5lZHUiLCJmaXJzdF9uYW1lIjoiU2ViYXN0aWFuIiwibGFzdF9uYW1lIjoiR3JhbW1hcyIsInByb3BlcnRpZXMiOnsibWV0YWRhdGEiOnsiZGJfdXNlcl9pZCI6MzQxN319fQ.SQWfR5occffJw1W1pJiA7S8KxQF2A_qDB2p13ydJukZJarH83fIc1NLhm3VA5EzySVydjDv1vxcSm3prsedmqaZL4EbAYy08R5goJAMzUIEp8-R7WPNEDaLyQd5AenFE2zlaWAA5y2Gqx3216_1Ltfgp3bLEPkeys6oDWcjix40r_kdj46iafXsC3Yts5lY_Cmav1XR8M6LUvWubQom-Rl61OsVpknpVfBaI_Z8aT9nmiSyLJoohc_-MD5HA3g-F65oVb2I3JGJTeje98PEdXzBlVMEI7TzZZ2uFMmTApYUDYfCneMWYpz-Abe-Yf1xaSYevmQOyzGDtGRkrjyW2dQ",
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
    "body": "{\"query_type\":\"single_company\",\"limit\":50,\"slug\":\"the-farmers-dog-f1cda844-ebdf-4437-a0b7-38522957a770\",\"since\":\"all\",\"skip\":0}",
    "method": "POST"
  });