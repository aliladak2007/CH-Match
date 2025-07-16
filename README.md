# Companies House Matcher

A Streamlit web application that matches company names from an Excel file against the UK Companies House database, retrieving comprehensive company information including registration details, officers, and financial data.

## Features

- **Bulk Company Matching**: Upload Excel files with company names and get automated matches
- **Intelligent Matching**: Uses multiple search strategies including exact matching, suffix stripping, and fuzzy matching
- **Comprehensive Data**: Returns company numbers, addresses, officers, status, and financial information
- **High Performance**: Concurrent processing for large datasets with rate limiting
- **Fallback Search**: Google fallback search for companies not found via direct API
- **Excel Export**: Download results as formatted Excel files

## What Information is Retrieved

For each company, the tool attempts to find:

- **Company Number**: Official UK company registration number
- **Official Name**: Registered company name
- **Registered Address**: Full registered office address
- **Company Status**: Active, dissolved, liquidation, etc.
- **Company Type**: Private limited, PLC, etc.
- **Active Officers**: Current directors and secretaries with dates of birth
- **Financial Data**: Last accounts date and next due date
- **Match Quality**: Similarity score and exact match indicators

## Prerequisites

- Python 3.8 or higher
- Companies House API key (free from [Companies House Developer Hub](https://developer.company-information.service.gov.uk/))
- Required Python packages (see Installation)

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/aliladak2007/companies-house-matcher.git
cd companies-house-matcher
```

2. **Install required packages**:
```bash
pip install streamlit pandas requests aiohttp openpyxl duckduckgo-search
```

3. **Set up your API key**:

   **Option 1: Environment Variable (Recommended)**
   ```bash
   export COMPANIES_HOUSE_API_KEY="your_api_key_here"
   ```

   **Option 2: Streamlit Secrets**
   Create `.streamlit/secrets.toml`:
   ```toml
   COMPANIES_HOUSE_API_KEY = "your_api_key_here"
   ```

   **Option 3: Direct Code Edit**
   Edit line 26 in `CHTool.py`:
   ```python
   os.environ["COMPANIES_HOUSE_API_KEY"] = "your_actual_api_key_here"
   ```

## Usage

1. **Start the application**:
```bash
streamlit run CHTool.py
```

2. **Prepare your Excel file**:
   - Must contain a column named `CompanyName`
   - Optionally include a `CompanyNumber` column for direct lookups
   - Example format:
   
   | CompanyName | CompanyNumber |
   |-------------|---------------|
   | Apple Ltd   |               |
   | Microsoft UK| 12345678      |
   | Google Inc  |               |

3. **Upload and process**:
   - Upload your Excel file via the web interface
   - Click "Run Matching" to start processing
   - Monitor progress via the progress bar
   - Download results when complete

## Input File Format

### Required Column
- `CompanyName`: Company names to search for

### Optional Column
- `CompanyNumber`: If provided, the tool will:
  - Look up company by number first
  - Verify the name matches (90%+ similarity required)
  - Fall back to name search if number lookup fails

### Example Input
```csv
CompanyName,CompanyNumber
"Apple Retail UK Limited",
"Microsoft Limited",12345678
"Google UK Ltd",
"Amazon UK Services Ltd",
```

## Output Format

The results include your original data plus these additional columns:

| Column | Description |
|--------|-------------|
| `company_number` | Official UK company number |
| `official_name` | Registered company name |
| `similarity_score` | Match quality (0.0-1.0) |
| `possible_matches` | Alternative matches found |
| `exact_match` | Yes/No indicator with match method |
| `registered_address` | Full registered office address |
| `registered_officers` | Active directors and secretaries |
| `company_status` | Current company status |
| `company_type` | Legal entity type |
| `last_made_up_to` | Last accounts filing date |
| `next_accounts_due_on` | Next accounts due date |

## Search Strategy

The tool uses a sophisticated multi-stage search approach:

1. **Direct Number Lookup**: If company number provided, look up directly
2. **Exact Name Match**: Search for the exact company name
3. **Suffix Stripping**: Remove common suffixes (Ltd, Limited, PLC, etc.)
4. **Word Truncation**: Try first 2-3 words of company name
5. **Quoted Search**: Exact phrase matching
6. **Google Fallback**: Web search as last resort (limited to 5 per session)

## Rate Limiting and Performance

- **Concurrent Processing**: Uses async/await for better performance
- **Rate Limiting**: Built-in delays to respect API limits
- **Error Handling**: Graceful handling of API errors and timeouts
- **Progress Tracking**: Real-time progress updates

## Troubleshooting

### Common Issues

**"No Companies House API key supplied"**
- Ensure your API key is properly set via environment variable or secrets
- Check that the key is valid and active

**"Rate-limited fetching profile"**
- The tool handles rate limiting automatically
- Large files may take longer due to API limits

**"No matches found"**
- Try variations of the company name
- Check for typos or alternative spellings
- Some companies may not be in the UK database

**Excel upload errors**
- Ensure file has `.xlsx` extension
- Check that `CompanyName` column exists
- Verify file isn't corrupted

### Performance Tips

- **Batch Processing**: Process large files during off-peak hours
- **Clean Data**: Remove duplicates and clean company names before upload
- **API Limits**: Be aware of daily API rate limits
- **Internet Connection**: Ensure stable internet for concurrent processing

## API Rate Limits

Companies House API has the following limits:
- 600 requests per 5-minute window
- The tool automatically handles rate limiting with delays and retries

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Companies House](https://www.gov.uk/government/organisations/companies-house) for providing the API
- [Streamlit](https://streamlit.io/) for the web framework
- [DuckDuckGo](https://duckduckgo.com/) for fallback search capabilities

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing issues for solutions
- Provide detailed error messages and steps to reproduce

---

⚠️ **Important**: This tool is for legitimate business research only. Respect rate limits and use responsibly. The Companies House API terms of service apply to all usage.
