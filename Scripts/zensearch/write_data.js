import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";

// Get script directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Define paths
const DIRECTORY = path.resolve(__dirname, "../../zensearchData");
if (!fs.existsSync(DIRECTORY)) {
    fs.mkdirSync(DIRECTORY, { recursive: true });
}

const filePath = path.join(DIRECTORY, "company_data.csv");

// Function to check if the company already exists
function companyExists(companyName) {
    if (!fs.existsSync(filePath)) return false; // If file doesn't exist, company can't exist

    const fileContent = fs.readFileSync(filePath, "utf8");
    const lines = fileContent.split("\n").map(line => line.trim());

    return lines.some(line => {
        const columns = line.split(",");
        return columns[0]?.toLowerCase() === companyName.toLowerCase(); // Case-insensitive match
    });
}

// Function to prompt user for company data
async function getCompanyData() {
    const { default: promptSync } = await import("prompt-sync");
    const prompt = promptSync();
    
    let companyName;
    do {
        companyName = prompt("Enter Company Name: ").trim();
        if (companyExists(companyName)) {
            console.log(`❌ Company "${companyName}" already exists. Please enter a different name.`);
        }
    } while (companyExists(companyName)); // Keep asking if company already exists

    const slug = prompt("Enter Slug: ").trim();
    const authToken = prompt("Enter Authorization Token: ").trim();
    
    return { companyName, slug, authToken };
}

// Function to write to CSV
function writeToCSV(companyData) {
    const dataRow = `${companyData.companyName},${companyData.slug},${companyData.authToken}\n`;

    // Check if file exists to add headers
    if (!fs.existsSync(filePath)) {
        const headers = "Company Name,Slug,Authorization Token\n";
        fs.writeFileSync(filePath, headers, "utf8");
    }

    // Append new data
    fs.appendFileSync(filePath, dataRow, "utf8");
    console.log(`✅ Data written successfully to ${filePath}`);
}

// Main execution
(async () => {
    const newCompany = await getCompanyData();
    writeToCSV(newCompany);
})();
