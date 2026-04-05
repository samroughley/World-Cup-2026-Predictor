from playwright.sync_api import sync_playwright
import pandas as pd

def scrape_opta_live():
    url = "https://theanalyst.com/competition/fifa-world-cup/predictions"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Use a real browser header to avoid being blocked
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        page = context.new_page()
        
        print("Loading page and waiting for JS to fill the table...")
        page.goto(url, wait_until="domcontentloaded")

        # THE KEY STEP: Wait for a cell containing a '%' to appear.
        # This ensures the numbers are loaded before we scrape.
        try:
            page.wait_for_selector("td:has-text('%')", timeout=15000)
        except:
            print("Timed out waiting for numbers. The table might be in an iframe.")
            # Fallback: search all frames if the main page doesn't have it
            for frame in page.frames:
                if frame.locator("td:has-text('%')").count() > 0:
                    page = frame
                    break

        # Now that we know it's filled, grab all rows with data-rankstatus
        rows = page.locator("tr[data-rankstatus]").all()
        
        scraped_data = []
        for row in rows:
            # Get the team name (first cell) and all number cells
            team = row.locator("td").first.inner_text().strip()
            # Find all cells in this row that contain a '%'
            probs = row.locator("td:has-text('%')").all_text_contents()
            
            if team and probs:
                scraped_data.append([team] + [p.replace('%', '') for p in probs])

        browser.close()
        
        # Create DataFrame
        cols = ['Team', 'R32_Exit', 'R16', 'QF', 'SF', 'Final', 'Winner']
        df = pd.DataFrame(scraped_data, columns=cols)
        
        # Convert numbers to floats for comparison
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col])
            
        return df

# Run it
opta_df = scrape_opta_live()
if not opta_df.empty:
    print(opta_df.sort_values("Winner", ascending=False).head(10))

    # Save the predictions
    opta_df.to_csv('Opta_Predictions.csv', index=False)