const fs = require('fs');
const path = require('path');
const prompt = require('prompt-sync')();

// Define CSV file path
const filePath = "C:\\Users\\agabr\\CareerMatch-AI\\zensearchData\\company_data.csv";

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
    console.log('✅ Data written successfully!');
}

// Main execution
const newCompany = getCompanyData();
writeToCSV(newCompany);
