import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";

// Get script directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Define paths
const DIRECTORY = path.resolve(__dirname, "../../zensearchData"); // Ensure correct path
if (!fs.existsSync(DIRECTORY)) {
    fs.mkdirSync(DIRECTORY, { recursive: true });
}

const filePath = path.join(DIRECTORY, "company_data.csv");

// Function to prompt user for company data (using dynamic import)
async function getCompanyData() {
    const { default: promptSync } = await import("prompt-sync");
    const prompt = promptSync();
    
    const companyName = prompt("Enter Company Name: ");
    const slug = prompt("Enter Slug: ");
    const authToken = prompt("Enter Authorization Token: ");
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
