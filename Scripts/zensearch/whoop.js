fetch("https://api.zensearch.jobs/api/postings", {
    "headers": {
      "accept": "*/*",
      "accept-language": "en-US,en;q=0.9",
      "authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImQ5ZTVmMmY0LTYxYjctNDY1OS1iYThmLWY4YTM0MjFiNTMyYyJ9.eyJzdWIiOiI4ZjQ2MDJlYy05MTkyLTRmYmItOWEzYi1iMWVkNmVkYzBkZmQiLCJpYXQiOjE3MzgyNTI1MTQsImV4cCI6MTczODI1NDMxNCwidXNlcl9pZCI6IjhmNDYwMmVjLTkxOTItNGZiYi05YTNiLWIxZWQ2ZWRjMGRmZCIsImlzcyI6Imh0dHBzOi8vYXV0aC56ZW5zZWFyY2guam9icyIsImVtYWlsIjoiZ3JhbW1zQHJwaS5lZHUiLCJmaXJzdF9uYW1lIjoiU2ViYXN0aWFuIiwibGFzdF9uYW1lIjoiR3JhbW1hcyIsInByb3BlcnRpZXMiOnsibWV0YWRhdGEiOnsiZGJfdXNlcl9pZCI6MzQxN319fQ.aZAHJu3wMtlLo_kRjLaUufOCXL-VSPGwI2ryr7O4UKVlRESZXvv8S4MVbFCB8aWwxkwoYsA68mxW52EB6atPDI9LQEfwynDlUAMrYzqJsIhAiaw868lzl9tWxqiylZsGNWhBzx5f4wiUNkDwFM3BpSJJfF50c62KoDgDPJ6ZIkz_Pm4_En-AB0dABX4WFdlrLwJLWNQ-IX3t1ZrSb_jBkffRP2OEN1nCquYCluP2sy_qqLXQqAe-1WlmeWuPChFHptQ15dH6q54Uuk5XnHlg1zPGABC5prxVW95mXW_OFnlV2iZAygU1dvOGfG0KliUlK6H0jB4Q7JXBl38cKsbriw",
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
    "body": "{\"query_type\":\"single_company\",\"limit\":50,\"slug\":\"whoop-48623dfe-4efe-4a07-bbd8-3d6423a1c74a\",\"since\":\"all\",\"skip\":0}",
    "method": "POST"
  });