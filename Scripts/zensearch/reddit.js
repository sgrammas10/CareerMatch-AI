fetch("https://api.zensearch.jobs/api/postings", {
    "headers": {
      "accept": "*/*",
      "accept-language": "en-US,en;q=0.9",
      "authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImQ5ZTVmMmY0LTYxYjctNDY1OS1iYThmLWY4YTM0MjFiNTMyYyJ9.eyJzdWIiOiI4ZjQ2MDJlYy05MTkyLTRmYmItOWEzYi1iMWVkNmVkYzBkZmQiLCJpYXQiOjE3MzgyNTI1NjcsImV4cCI6MTczODI1NDM2NywidXNlcl9pZCI6IjhmNDYwMmVjLTkxOTItNGZiYi05YTNiLWIxZWQ2ZWRjMGRmZCIsImlzcyI6Imh0dHBzOi8vYXV0aC56ZW5zZWFyY2guam9icyIsImVtYWlsIjoiZ3JhbW1zQHJwaS5lZHUiLCJmaXJzdF9uYW1lIjoiU2ViYXN0aWFuIiwibGFzdF9uYW1lIjoiR3JhbW1hcyIsInByb3BlcnRpZXMiOnsibWV0YWRhdGEiOnsiZGJfdXNlcl9pZCI6MzQxN319fQ.AUnfDO6Acyz8F-fR7dVXBvfVVk-R2ohGR04-rO70INzsfvI7fGW_elZQfo0yPzkBk8yOM9IJhH5kyUutaLFOR0yUcNcUJuIKTW1Ud61EUylF3UU4ubHLByQEsnFxL0jHb78YbGstOOmy5_qr9ayipT9cDUcOVrgIUNfQvgFe2_fhqD7dl0J033PXsrZLkcXxH8MvPiLzrluEk6NkP3gzX1sLvgGgAvv_xvyceXRjbpGgSNKxwEhLDMpwyW6JlQckCrFM48AIMRqm0HRRCte3vr5CSAUHVPTAGMSZmW1j7fuJ6eoolvn0ImbOxHyhNYuzw3qVggvoxLZTu3yXcTgjPw",
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
    "body": "{\"query_type\":\"single_company\",\"limit\":50,\"slug\":\"reddit-dad09ab1-e759-4b0a-8332-00ddf322ac81\",\"since\":\"all\",\"skip\":0}",
    "method": "POST"
  });