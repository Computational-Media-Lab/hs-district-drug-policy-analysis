import os
import re
import time
import random
import logging
import shutil
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


class TASBPolicyScraper:

    def __init__(self):
        self.config = {
            "base_url": "https://pol.tasb.org/PolicyOnline/PolicyDetails?key=",
            "districts": {"DISTRICT-NAME": "DISTRICT-CODE"}, # Information redacted due to data privacy
            "output_dir": "data",
            "download_dir": os.path.abspath("downloads"),
            "timeout": 30,
            "page_delay": 2
        }

        os.makedirs(self.config["download_dir"], exist_ok=True)
        self.driver = None
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def create_driver(self):
        chrome_options = Options()
        chrome_options.headless = False  # Downloads safely require a visual/non-headless layout context in many systems

        prefs = {
            "download.default_directory": self.config["download_dir"],
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--window-size=1920,1080")

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.logger.info("WebDriver initialized")

    def ensure_folder(self, *paths):
        path = os.path.join(*paths)
        os.makedirs(path, exist_ok=True)
        return path

    def smart_delay(self):
        time.sleep(self.config["page_delay"] + random.uniform(0.5, 1.5))

    def load_excel(self, excel_path):
        xls = pd.ExcelFile(excel_path)
        return {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}

    # ---------------- URL BUILDING ---------------- #

    def build_policy_url(self, district_key, doc_type, policy_type):
        """Builds standard front-facing UI link."""
        # Maps internal types to the specific hash fragments TASB expects
        hash_map = {
            'local': 'localTabContent',
            'legal': 'legalTabContent',
            'regulation': 'regulationsTabContent',
            'exhibit': 'exhibitTabContent'
        }
        tab = hash_map.get(policy_type.lower())
        return f"{self.config['base_url']}{district_key}&code={doc_type}#{tab}"

    def build_direct_download_url(self, district_key, doc_type, policy_type):
        """
        Constructs the precise backend file stream endpoint for all 4 variants.
        """
        type_backend_map = {
            'local': 'LOCAL',
            'legal': 'LEGAL',
            'regulation': 'REGULATION',
            'exhibit': 'XHIBIT'
        }
        backend_suffix = type_backend_map.get(policy_type, policy_type.upper())
        # If this direct link structure continues to struggle with LOCAL on their CDN,
        # our enhanced fallback state machine below will handle it seamlessly.
        return f"https://pol.tasb.org/Policy/Download/{district_key}?filename={doc_type}({backend_suffix}).pdf"

    # ---------------- DYNAMIC UI CLICKER ---------------- #

    def click_pdf_button(self, policy_type):
        """
        Locates and clicks any valid TASB PDF download button variation:
        dlpPDF (Local), dlfPDF (Legal), drPDF (Regulation), or dePDF (Exhibit).
        """
        # Map policy types to their exact element ID attributes on the TASB platform
        id_map = {
            'local': 'dlpPDF',
            'legal': 'dlfPDF',
            'regulation': 'drPDF',
            'exhibit': 'dePDF'
        }

        target_id = id_map.get(policy_type.lower())

        if not target_id:
            self.logger.warning(f"Unrecognized policy type '{policy_type}'. Could not map to a button ID.")
            return False

        try:
            # FIX: By.ID is cleaner here, but if you want By.CSS_SELECTOR use f"#{target_id}"
            pdf_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.ID, target_id))
            )
            
            try:
                pdf_button.click()
            except Exception:
                # Fallback if the element is intercepted or hidden behind a header/sticky banner
                self.driver.execute_script("arguments[0].click();", pdf_button)
            
            self.logger.info(f"Successfully triggered UI button download for variant: {target_id}")
            return True
        except Exception as e:
            self.logger.warning(f"Could not locate PDF button variant ({target_id}) on this page: {e}")
            return False

    # =========================================================================
    # FILE CHECKERS
    # =========================================================================
    def wait_for_download(self, timeout=30):
        start = time.time()
        while time.time() - start < timeout:
            files = [f for f in os.listdir(self.config["download_dir"]) if f.lower().endswith(".pdf")]
            if files:
                return True
            time.sleep(1)
        return False

    def get_latest_pdf(self):
        pdfs = [
            os.path.join(self.config["download_dir"], f)
            for f in os.listdir(self.config["download_dir"])
            if f.lower().endswith(".pdf")
        ]
        return max(pdfs, key=os.path.getctime) if pdfs else None

    # =========================================================================
    # CORE PIPELINE PIPING
    # =========================================================================
    def process_policy_download(self, text, district_key, doc_type, policy_type, category_folder):
        """
        Processes the workflow by cleanly transitioning from direct route mapping 
        to UI clicking using your updated explicit argument tracker.
        """
        try:
            self.driver.current_url
        except Exception:
            self.logger.warning("Session dropped. Re-instantiating WebDriver context...")
            self.create_driver()

        # Clean workspace to prevent false positives
        for f in os.listdir(self.config["download_dir"]):
            if f.lower().endswith((".pdf", ".crdownload", ".tmp")):
                try: os.remove(os.path.join(self.config["download_dir"], f))
                except Exception: pass

        download_triggered = False

        # --- Approach 1: Variant Direct Path ---
        try:
            direct_url = self.build_direct_download_url(district_key, doc_type, policy_type)
            self.logger.info(f"Attempting Variant Direct Download [{policy_type.upper()}]: {direct_url}")
            self.driver.get(direct_url)
            
            # Short timeout check to ensure we don't hang if the endpoint is dead
            if self.wait_for_download(timeout=4):
                downloaded = self.get_latest_pdf()
                if downloaded and os.path.getsize(downloaded) > 1024:
                    download_triggered = True
                else:
                    if downloaded: os.remove(downloaded)
        except Exception:
            pass

        # --- Approach 2: UI Tab View & Button Fallback ---
        if not download_triggered:
            standard_url = self.build_policy_url(district_key, doc_type, policy_type)
            self.logger.info(f"Direct route empty/failed. Navigating to layout UI page: {standard_url}")
            
            self.driver.get(standard_url)
            self.smart_delay()
            
            # FIX: Explicitly pass the policy_type string argument to your new click function!
            download_triggered = self.click_pdf_button(policy_type)

        # --- Move & Rename File Verification ---
        if download_triggered and self.wait_for_download(timeout=15):
            downloaded = self.get_latest_pdf()
            
            if downloaded and os.path.getsize(downloaded) > 1024:
                safe_name = re.sub(r"[^\w\-]+", "_", f"{text}_{policy_type.upper()}")
                final_path = os.path.join(category_folder, f"{safe_name}.pdf")
                shutil.move(downloaded, final_path)
                self.logger.info(f"Successfully Moved & Saved: {final_path}")
                return True
            else:
                self.logger.warning(f"File caught for '{text}' was corrupted or an error document template.")
                if downloaded:
                    try: os.remove(downloaded)
                    except Exception: pass
        else:
            self.logger.info(f"Document variant context unavailable or unconfigured on TASB server for: {text}")
        
        return False
    # =========================================================================
    # RUNNER ROUTINE
    # =========================================================================
    def scrape_from_excel(self, excel_path):
        district_data = self.load_excel(excel_path)
        self.create_driver()

        try:
            for district, df in district_data.items():
                districts_omit = ['Austin ISD', 'Amarillo ISD', 'Clear Creek ISD', 'Huffman ISD', 
                                'La Feria ISD', 'Longview ISD', 'San Isidro ISD', 'Terrell ISD', 
                                'Libery-Eylau ISD', 'Fabens ISD']

                if district in districts_omit:
                    continue

                if district not in self.config["districts"]:
                    self.logger.warning(f"District configuration mapping missing for: {district}")
                    continue

                district_key = self.config["districts"][district]
                district_folder = self.ensure_folder(self.config["output_dir"], district)
                self.logger.info(f"Processing District: {district}")

                for column in df.columns:
                    category_folder = self.ensure_folder(district_folder, column)

                    for cell in df[column].dropna():
                        text = str(cell).strip()
                        if not text or "(" not in text:
                            continue

                        doc_type = text.split("(")[0].strip()
                        policy_type = text[text.find("(")+1:text.find(")")].lower()

                        if policy_type in ['legal', 'local', 'regulation', 'exhibit']:
                            self.process_policy_download(text, district_key, doc_type, policy_type, category_folder)
                            self.smart_delay()
                        else:
                            self.logger.warning(f"Unsupported policy format type found: {policy_type}")

        finally:
            if self.driver:
                self.driver.quit()
            self.logger.info("Scraping workflow completed execution run.")



if __name__ == "__main__":
    scraper = TASBPolicyScraper()
    scraper.scrape_from_excel("Texas_School_District_Policies.xlsx")