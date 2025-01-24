import fetch from "node-fetch"

//Example fetch function
//Fetches from https://www.itftennis.com/en/tournament-calendar/mens-world-tennis-tour-calendar/?categories=All&startdate=2025-01

async function fetchTournamentData() {
    try {
      const response = await fetch("https://www.itftennis.com/tennis/api/TournamentApi/GetCalendar?circuitCode=MT&searchString=&skip=0&take=100&nationCodes=&zoneCodes=&dateFrom=2025-01-01&dateTo=2025-01-31&indoorOutdoor=&categories=&isOrderAscending=true&orderField=startDate&surfaceCodes=", {
        "headers": {
          "accept": "*/*",
          "accept-language": "en-US,en;q=0.9",
          "if-modified-since": "Fri, 24 Jan 2025 19:57:13 GMT",
          "priority": "u=1, i",
          "sec-ch-ua": "\"Google Chrome\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
          "sec-ch-ua-mobile": "?0",
          "sec-ch-ua-platform": "\"macOS\"",
          "sec-fetch-dest": "empty",
          "sec-fetch-mode": "cors",
          "sec-fetch-site": "same-origin",
          "cookie": "ARRAffinity=1a7aff82bc21373b03d8fda86d009014a254fb43661cd4068b45b28f7aa56160; ARRAffinitySameSite=1a7aff82bc21373b03d8fda86d009014a254fb43661cd4068b45b28f7aa56160; nlbi_178373=Ran5N41HZQkbS5zEtoSRdQAAAABom3lJWWsIWcVYqOPhRl1C; visid_incap_178373=7jzkGB/oTpq6/Jc/BRvGgbVdCGcAAAAAQUIPAAAAAADUwUyZ73WafusZyKj+p7zW; OptanonAlertBoxClosed=2024-10-10T23:05:34.074Z; incap_ses_182_178373=as+rMSL0V37QcjIjQZiGAtEBlGcAAAAAORKY7tNtvmc9UA8GrvugGw==; OptanonConsent=isGpcEnabled=0&datestamp=Fri+Jan+24+2025+16%3A11%3A15+GMT-0500+(Eastern+Standard+Time)&version=6.23.0&isIABGlobal=false&hosts=&consentId=a7629d4b-c3f6-4ed2-a3d9-6d1b9259adc2&interactionCount=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0002%3A1%2CC0003%3A1%2CC0004%3A1%2CC0005%3A1&geolocation=US%3BNY&AwaitingReconsent=false"
        },
        "referrerPolicy": "no-referrer",
        "body": null,
        "method": "GET"
      });
  
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
  
      const data = await response.json();
      return data;
    } catch (error) {
      console.error("Failed to fetch tournament data:", error);
      throw error;
    }
}
let dataOut = await fetchTournamentData()
console.log(dataOut)