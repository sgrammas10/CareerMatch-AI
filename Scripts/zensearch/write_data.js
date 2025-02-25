const fs = require('fs');
const path = require('path');
const prompt = require('prompt-sync')();

/**
 * Recursively searches for a file in a directory and its subdirectories.
 * @param {string} startPath - The directory to start searching from.
 * @param {string} fileName - The name of the file to search for.
 * @returns {string|null} - The full path to the file if found, otherwise null.
 */
function findFile(startPath, fileName) {
    if (!fs.existsSync(startPath)) return null;

    const files = fs.readdirSync(startPath);
    for (const file of files) {
        const filePath = path.join(startPath, file);
        const stat = fs.statSync(filePath);

        if (stat.isDirectory()) {
            const found = findFile(filePath, fileName);
            if (found) return found;
        } else if (file === fileName) {
            return filePath;
        }
    }
    return null;
}

// Find company_data.csv dynamically
const scriptDir = __dirname; // Get the directory where the script is located
const csvFileName = "company_data.csv";
let filePath = findFile(scriptDir, csvFileName);

// If not found, default to script directory
if (!filePath) {
    filePath = path.join(scriptDir, csvFileName);
    console.log(`📁 File not found, creating it in: ${filePath}`);
}

// Function to prompt user for company data
function getCompanyData() {
    const companyName = prompt('Enter Company Name: ');
    const slug = prompt('Enter Slug: ');
    const authToken = prompt('Enter Authorization Token: ');
    return { companyName, slug, authToken };
}

// Function to write to CSV
function writeToCSV(companyData) {
    const dataRow = `${companyData.companyName},${companyData.slug},${companyData.authToken}\n`;

    // Check if file exists to add headers
    if (!fs.existsSync(filePath)) {
        const headers = "Company Name,Slug,Authorization Token\n";
        fs.writeFileSync(filePath, headers, 'utf8');
    }

    // Append new data
    fs.appendFileSync(filePath, dataRow, 'utf8');
    console.log(`✅ Data written successfully to ${filePath}`);
}

// Main execution
const newCompany = getCompanyData();
writeToCSV(newCompany);
