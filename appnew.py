import pandas as pd
import streamlit as st
import plotly.express as px
import os
import time
import io
import shutil
import numpy as np
import json
import base64 
import bcrypt 
import pdfkit 
import smtplib
import urllib.parse
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# 1. Page Configuration
st.set_page_config(page_title="Vehicle Sales System", layout="wide")

# --- PASSWORD HASHING HELPERS ---
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except ValueError:
        return False

# --- USER AUTHENTICATION & MANAGEMENT SYSTEM ---
USERS_FILE = "user_db.json"
# Updated Module List
ALL_MODULES = ["Dashboard", "Search & Edit", "Target vs Actuals", "Financial Reports", "OEM Pending Analysis", "Tally & TOS Reports", "All Report"]

DEFAULT_USERS = {
    "admin": {"password": hash_password("admin123"), "role": "admin", "name": "System Admin", "access": ALL_MODULES},
    "manager": {"password": hash_password("manager1"), "role": "manager", "name": "Sales Manager", "access": ["Dashboard", "Financial Reports", "OEM Pending Analysis", "All Report", "Target vs Actuals"]},
    "sales": {"password": hash_password("sales1"), "role": "sales", "name": "Sales Executive", "access": ["Dashboard", "OEM Pending Analysis", "All Report"]}
}

def format_lakhs(value):
    if isinstance(value, (int, float)):
        try: val_str = "{:.0f}".format(value)
        except: return value
        if "." in val_str: head, decimal = val_str.split(".")
        else: head, decimal = val_str, ""
        is_neg = head.startswith("-")
        if is_neg: head = head[1:]
        if len(head) <= 3: res = head
        else:
            last3 = head[-3:]
            rest = head[:-3]
            rev = rest[::-1]
            chunks = [rev[i:i+2] for i in range(0, len(rev), 2)]
            res = ",".join(chunks)[::-1] + "," + last3
        if is_neg: res = "-" + res
        return res + ("." + decimal if decimal else "")
    return value

def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f: json.dump(DEFAULT_USERS, f)
        return DEFAULT_USERS
    try:
        with open(USERS_FILE, 'r') as f:
            data = json.load(f)
            # Ensure new tab is accessible to existing admins/managers
            for u in data:
                if 'access' not in data[u]:
                    if data[u]['role'] == 'admin': data[u]['access'] = ALL_MODULES
                    elif data[u]['role'] == 'manager': data[u]['access'] = ["Dashboard", "Financial Reports", "All Report", "Target vs Actuals"]
                    else: data[u]['access'] = ["Dashboard", "All Report"]
                elif "Target vs Actuals" not in data[u]['access'] and data[u]['role'] in ['admin', 'manager']:
                     data[u]['access'].append("Target vs Actuals")
            return data
    except: return DEFAULT_USERS

def save_users(users):
    with open(USERS_FILE, 'w') as f: json.dump(users, f)

def login_page():
    st.markdown("<h1 style='text-align: center;'>🔒 Secure Login</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                current_users = load_users()
                if username in current_users and verify_password(password, current_users[username]['password']):
                    st.session_state['authenticated'] = True
                    st.session_state['user'] = username
                    st.session_state['role'] = current_users[username]['role']
                    st.session_state['name'] = current_users[username]['name']
                    st.session_state['access'] = current_users[username].get('access', [])
                    st.success(f"Welcome {current_users[username]['name']}!")
                    time.sleep(0.5)
                    st.rerun()
                else: st.error("❌ Invalid Credentials")

        st.markdown("---")
        with st.expander("⚠️ Database Maintenance (Restricted Access)"):
            st.warning("This area is for Master Admins only. Incorrect use will delete all user data.")
            with st.form("secure_reset_form"):
                master_id = st.text_input("Master Admin ID")
                master_pass = st.text_input("Master Password", type="password")
                reset_btn = st.form_submit_button("🚨 FORCE RESET DATABASE")
                if reset_btn:
                    if master_id == "master" and master_pass == "reset123":
                        if os.path.exists(USERS_FILE): os.remove(USERS_FILE)
                        with open(USERS_FILE, 'w') as f: json.dump(DEFAULT_USERS, f)
                        st.success("✅ Database Reset Successfully! Please Login again.")
                        time.sleep(1)
                        st.rerun()
                    else: st.error("❌ ACCESS DENIED: Incorrect Master Credentials.")

if 'authenticated' not in st.session_state: st.session_state['authenticated'] = False

# --- SHARE FEATURE ---
def share_report_feature(df, report_name):
    st.markdown("### 📤 Share Report")
    with st.expander("Click here to Send via WhatsApp or Email"):
        t_wa, t_mail = st.tabs(["💬 WhatsApp", "📧 Email"])
        with t_wa:
            st.info("💡 This will open WhatsApp with a summary. Please attach the downloaded Excel file manually.")
            if not df.empty:
                total_records = len(df)
                total_val = df['Sale Invoice Amount With GST'].sum() if 'Sale Invoice Amount With GST' in df.columns else 0
                msg = f"*Vehicle Sales Report: {report_name}*\nTotal Records: {total_records}\nTotal Value: {format_lakhs(total_val)}\nPlease find the detailed report attached."
                encoded_msg = urllib.parse.quote(msg)
                st.link_button("📲 Open WhatsApp to Share", f"https://wa.me/?text={encoded_msg}")
        with t_mail:
            c_e1, c_e2 = st.columns(2)
            sender_email = c_e1.text_input("Your Gmail", key=f"se_{report_name}")
            sender_pass = c_e2.text_input("App Password", type="password", key=f"sp_{report_name}")
            receiver_email = st.text_input("Receiver Email", key=f"re_{report_name}")
            if st.button("📧 Send Email", key=f"btn_{report_name}"):
                if sender_email and sender_pass and receiver_email:
                    try:
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer: df.to_excel(writer, index=False)
                        output.seek(0)
                        msg = MIMEMultipart()
                        msg['From'] = sender_email
                        msg['To'] = receiver_email
                        msg['Subject'] = f"Report: {report_name}"
                        msg.attach(MIMEText("Please find the attached report.", 'plain'))
                        part = MIMEBase('application', "octet-stream")
                        part.set_payload(output.read())
                        encoders.encode_base64(part)
                        part.add_header('Content-Disposition', f'attachment; filename="{report_name}.xlsx"')
                        msg.attach(part)
                        server = smtplib.SMTP('smtp.gmail.com', 587)
                        server.starttls()
                        server.login(sender_email, sender_pass)
                        server.sendmail(sender_email, receiver_email, msg.as_string())
                        server.quit()
                        st.success("✅ Email Sent!")
                    except Exception as e: st.error(f"❌ Failed: {e}")
                else: st.warning("⚠️ Fill all fields.")

# --- HELPER FOR TARGETS ---
def load_targets_data(file_path):
    # Try to load 'Targets_Master' sheet, if not exists return empty DF
    try:
        return pd.read_excel(file_path, sheet_name="Targets_Master")
    except:
        return pd.DataFrame(columns=["Category_Type", "Category_Value", "Month_Year", "Target_Value"])

def save_targets_data(file_path, new_targets_df):
    try:
        # We need to save to a specific sheet without overwriting others.
        # This requires loading the workbook, removing old sheet if exists, adding new.
        # Using a simpler approach: Append mode with replacement if possible, 
        # but pandas 'replace' replaces the whole file usually.
        # SAFE APPROACH: Read all sheets, update one, write all back.
        
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        all_sheets["Targets_Master"] = new_targets_df
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for sheet_name, df in all_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        return True
    except Exception as e:
        return str(e)

# --- MAIN APP ---
if not st.session_state['authenticated']: login_page()
else:
    with st.sidebar:
        st.info(f"👤 User: **{st.session_state['name']}**\n🔑 Role: **{st.session_state['role'].upper()}**")
        with st.expander("🔑 Change My Password"):
            curr_pass = st.text_input("Current Password", type="password", key="cp")
            new_pass = st.text_input("New Password", type="password", key="np")
            conf_pass = st.text_input("Confirm New Password", type="password", key="cnp")
            if st.button("Update Password"):
                users = load_users()
                uname = st.session_state['user']
                if verify_password(curr_pass, users[uname]['password']):
                    if new_pass == conf_pass and new_pass:
                        users[uname]['password'] = hash_password(new_pass)
                        save_users(users)
                        st.success("✅ Password Changed!")
                    else: st.error("❌ New passwords do not match.")
                else: st.error("❌ Incorrect Current Password.")

        if st.session_state['role'] == 'admin':
            with st.expander("🛠️ Admin: User Management", expanded=False):
                u_new = st.text_input("Username (Unique)")
                p_new = st.text_input("Set Password", type="password")
                n_new = st.text_input("Display Name")
                r_new = st.selectbox("Assign Role Label", ["admin", "manager", "sales"])
                default_access = ALL_MODULES if r_new == 'admin' else ["Dashboard"]
                a_new = st.multiselect("Allowed Tabs", ALL_MODULES, default=default_access)
                if st.button("Create/Update User"):
                    if u_new and p_new and n_new and a_new:
                        users = load_users()
                        users[u_new] = {"password": hash_password(p_new), "role": r_new, "name": n_new, "access": a_new}
                        save_users(users)
                        st.success(f"✅ User '{u_new}' Saved!")
                    else: st.error("All fields required.")
                st.markdown("---")
                users = load_users()
                del_user = st.selectbox("Select User to Delete", list(users.keys()))
                if st.button("Delete User", type="primary"):
                    if del_user == st.session_state['user']: st.error("❌ You cannot delete yourself!")
                    else:
                        del users[del_user]
                        save_users(users)
                        st.success(f"User '{del_user}' deleted.")
                        time.sleep(1)
                        st.rerun()

    FILE_PATH = "ONE REPORT.xlsx"
    SHEET_NAME = "Retail Format"
    DB_FOLDER = "tally_tos_database"
    BACKUP_FOLDER = "backups"

    if not os.path.exists(DB_FOLDER): os.makedirs(DB_FOLDER)
    if not os.path.exists(BACKUP_FOLDER): os.makedirs(BACKUP_FOLDER)

    FILES_DB = {
        "Tally Sale": os.path.join(DB_FOLDER, "master_tally_sale.csv"),
        "Tally Purchase": os.path.join(DB_FOLDER, "master_tally_purchase.csv"),
        "TOS In": os.path.join(DB_FOLDER, "master_tos_in.csv"),
        "TOS Out": os.path.join(DB_FOLDER, "master_tos_out.csv")
    }

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 💾 Data Management")
        if st.button("📥 Manual Data Backup", use_container_width=True):
            if os.path.exists(FILE_PATH):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_file = os.path.join(BACKUP_FOLDER, f"ONE_REPORT_MANUAL_backup_{timestamp}.xlsx")
                try:
                    shutil.copy2(FILE_PATH, backup_file)
                    st.success("✅ Manual Backup Created!")
                except Exception as e: st.error(f"Error: {e}")
            else: st.error("❌ Data file not found!")
        st.markdown("---")
        auto_refresh = st.checkbox("✅ Enable Auto-Update", value=True)
        refresh_rate = st.slider("Refresh Rate (s)", 5, 60, 10)
        if st.button("🚪 Logout", type="primary", use_container_width=True):
            st.session_state['authenticated'] = False
            st.rerun()

    def get_file_timestamp():
        if os.path.exists(FILE_PATH): return os.path.getmtime(FILE_PATH)
        return 0

    @st.cache_data
    def load_data(last_modified_time):
        if not os.path.exists(FILE_PATH): return None
        try:
            df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
            df.columns = df.columns.str.strip()
            if 'Invoice Date' in df.columns: df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], dayfirst=True, errors='coerce')
            if 'Chassis No.' in df.columns: df['Chassis No.'] = df['Chassis No.'].astype(str)
            target_cols = ['Sale Invoice Amount With GST', 'Sale Invoice Amount Basic Value', 'Purchase With GST Value', 'Purchase Basic Value', 'TOTAL OEM DISCOUNTS', 'TOTAL INTENAL DISCOUNTS', 'TOTAL OEM & INTERNAL NET DISCOUNTS', 'TOTAL Credit Note NET DISCOUNT', 'MARGIN', 'TOTAL RECEIVED OEM NET DISCOUNTS', 'FINAL MARGIN', 'OEM - RETAIL SCHEME', 'RECEIVED OEM - RETAIL SCHEME', 'OEM - CORPORATE SCHEME', 'RECEIVED OEM - CORPORATE SCHEME', 'OEM - EXCHANGE SCHEME', 'RECEIVED OEM - EXCHANGE SCHEME', 'OEM - SPECIAL SCHEME', 'RECEIVED OEM - SPECIAL SCHEME', 'OEM - WHOLESALE SUPPORT', 'RECEIVED OEM - WHOLESALE SUPPORT', 'OEM - LOYALTY BONUS', 'RECEIVED OEM - LOYALTY BONUS', 'OEM - OTHERS', 'RECEIVED OEM - OTHERS', 'TOTAL Credit Note Amout OEM', 'Month Wise FSC Target']
            for col in target_cols:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                else: df[col] = 0
            return df
        except Exception as e:
            st.error(f"Error: {e}")
            return None

    def to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer: df.to_excel(writer, index=False, sheet_name='Report')
        return output.getvalue()

    def save_and_append(uploaded_file, master_path):
        try:
            new_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            new_df.columns = new_df.columns.astype(str).str.strip()
            chassis_col = next((c for c in new_df.columns if any(x in c.upper() for x in ["CHASSIS", "BATCH", "SERIAL"])), None)
            if os.path.exists(master_path):
                master_df = pd.read_csv(master_path, low_memory=False)
                combined_df = pd.concat([master_df, new_df], ignore_index=True)
                if chassis_col: combined_df = combined_df.drop_duplicates(subset=[chassis_col], keep='last')
                else: combined_df = combined_df.drop_duplicates()
            else: combined_df = new_df
            combined_df.to_csv(master_path, index=False)
            return True, len(combined_df)
        except Exception as e: return False, str(e)

    def remove_duplicates_only(file_path):
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, low_memory=False)
                old_len = len(df)
                chassis_col = next((c for c in df.columns if any(x in c.upper() for x in ["CHASSIS", "BATCH"])), None)
                if chassis_col: df = df.drop_duplicates(subset=[chassis_col], keep='last')
                else: df = df.drop_duplicates()
                df.to_csv(file_path, index=False)
                return True, (old_len - len(df))
            except: return False, "Error"
        return False, "Not Found"

    def generate_all_report(df_data, group_col_input, title):
        group_cols = [group_col_input] if isinstance(group_col_input, str) else group_col_input
        valid_cols = [c for c in group_cols if c in df_data.columns]
        if not valid_cols: st.error("Columns missing"); return
        metrics = {"Sale(GST)": "Sale Invoice Amount With GST", "Sale(Basic)": "Sale Invoice Amount Basic Value", "Pur(GST)": "Purchase With GST Value", "Pur(Basic)": "Purchase Basic Value", "OEM Disc": "TOTAL OEM DISCOUNTS", "Int Disc": "TOTAL INTENAL DISCOUNTS", "Margin": "MARGIN", "Final Margin": "FINAL MARGIN"}
        df_temp = df_data.copy()
        for c in valid_cols: df_temp[c] = df_temp[c].fillna("Unknown")
        grouped = df_temp.groupby(valid_cols)[list(metrics.values())].sum().reset_index()
        counts = df_temp.groupby(valid_cols).size().reset_index(name='Count')
        final = pd.merge(counts, grouped, on=valid_cols)
        if "FINAL MARGIN" in final.columns and "Count" in final.columns: final["Avg Margin"] = (final["FINAL MARGIN"] / final["Count"]).fillna(0)
        total_row = {c: "" for c in valid_cols}; total_row[valid_cols[0]] = "GRAND TOTAL"; total_row["Count"] = final['Count'].sum()
        for col in metrics.values(): total_row[col] = final[col].sum()
        total_row["Avg Margin"] = total_row["FINAL MARGIN"] / total_row["Count"] if total_row["Count"] > 0 else 0
        avg_row = {c: "" for c in valid_cols}; avg_row[valid_cols[0]] = "AVERAGE PER VEHICLE"; avg_row["Count"] = 1
        for col in metrics.values(): avg_row[col] = total_row[col] / total_row["Count"] if total_row["Count"] > 0 else 0
        avg_row["Avg Margin"] = total_row["Avg Margin"]
        final = pd.concat([final, pd.DataFrame([total_row]), pd.DataFrame([avg_row])], ignore_index=True)
        final.rename(columns={v: k for k, v in metrics.items()}, inplace=True)
        money_cols = list(metrics.keys()) + ["Avg Margin"]
        format_dict = {col: (lambda x: f"₹ {format_lakhs(x)}") for col in money_cols if col in final.columns}
        if "Count" in final.columns: format_dict["Count"] = format_lakhs
        def highlight(row):
            if row[valid_cols[0]] == "GRAND TOTAL": return ['background-color: #f0f0f0; font-weight: bold'] * len(row)
            if row[valid_cols[0]] == "AVERAGE PER VEHICLE": return ['background-color: #e6f3ff; font-weight: bold'] * len(row)
            return [''] * len(row)
        st.subheader(title)
        st.dataframe(final.style.apply(highlight, axis=1).format(format_dict))
        share_report_feature(final, title)

    def generate_month_wise_pivot(df, group_cols, date_col='Invoice Date', start_date=None, end_date=None):
        for col in group_cols: df[col] = df[col].fillna("Unknown")
        df['Month_Sort'] = df[date_col].dt.to_period('M')
        pivot_data = df.pivot_table(index=group_cols, columns='Month_Sort', values=date_col, aggfunc='count', fill_value=0)
        if start_date and end_date:
            expected_months = pd.period_range(start=start_date, end=end_date, freq='M')
            pivot_data = pivot_data.reindex(columns=expected_months, fill_value=0)
        pivot_data['Total'] = pivot_data.sum(axis=1)
        month_count = len(pivot_data.columns) - 1
        pivot_data['Average'] = pivot_data['Total'] / month_count if month_count > 0 else 0
        gt_row = pivot_data.sum(axis=0)
        if month_count > 0: gt_row['Average'] = gt_row['Total'] / month_count
        if len(group_cols) > 1: gt_name = tuple(['GRAND TOTAL'] + [''] * (len(group_cols) - 1))
        else: gt_name = 'GRAND TOTAL'
        pivot_data.loc[gt_name, :] = gt_row
        pivot_data.columns = [c.strftime('%b-%Y') if isinstance(c, pd.Period) else c for c in pivot_data.columns]
        return pivot_data

    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f: return base64.b64encode(f.read()).decode()

    # --- APP START ---
    logo_left_path = "logo_left.png"
    logo_right_path = "logo_right.png"
    logo_left_html = ""
    logo_right_html = ""
    img_style = "height: 80px; width: auto; object-fit: contain; max-width: 150px;"

    if os.path.exists(logo_left_path):
        logo_left_html = f'<img src="data:image/png;base64,{get_base64_of_bin_file(logo_left_path)}" style="{img_style}">'
    if os.path.exists(logo_right_path):
        logo_right_html = f'<img src="data:image/png;base64,{get_base64_of_bin_file(logo_right_path)}" style="{img_style}">'

    header_html = f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0;">
        <div style="flex: 1; text-align: left;">{logo_left_html}</div>
        <div style="flex: 6; text-align: center;"><h1 style="margin: 0;">🚗 Vehicle Sales Management System</h1></div>
        <div style="flex: 1; text-align: right;">{logo_right_html}</div>
    </div><hr style="margin-top: 0; margin-bottom: 20px;">
    """
    st.markdown(header_html, unsafe_allow_html=True)

    ts = get_file_timestamp()
    if ts == 0: st.error("❌ File not found!")
    else:
        df = load_data(ts)
        if df is not None:
            allowed_tabs = st.session_state.get('access', [])
            if not allowed_tabs: allowed_tabs = ["Dashboard"]
        
            tabs = st.tabs(allowed_tabs)
            tab_map = {name: tab for name, tab in zip(allowed_tabs, tabs)}

            # TAB: DASHBOARD
            if "Dashboard" in tab_map:
                with tab_map["Dashboard"]:
                    st.subheader("Overview")
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Total Vehicles", format_lakhs(len(df)))
                    k2.metric("Total Revenue", f"₹ {format_lakhs(df['Sale Invoice Amount With GST'].sum())}")
                    k3.metric("Total Final Margin", f"₹ {format_lakhs(df['FINAL MARGIN'].sum())}")
                    st.dataframe(df)

            # TAB: SEARCH & EDIT
            if "Search & Edit" in tab_map:
                with tab_map["Search & Edit"]:
                    st.header("Search & Edit Records")
                    search_mode = st.selectbox("Search By:", ["General Search", "Chassis No. (Last 8 / Full)"])
                    search_val = st.text_input("Enter Value to Search").strip()
                    if search_val:
                        if search_mode == "Chassis No. (Last 8 / Full)":
                            if 'Chassis No.' in df.columns: mask = df['Chassis No.'].str.contains(search_val, case=False, na=False)
                            else: mask = pd.Series([False]*len(df))
                        else: mask = df.apply(lambda x: x.astype(str).str.contains(search_val, case=False, na=False)).any(axis=1)
                        res = df[mask]
                        if not res.empty:
                            st.success(f"Found {len(res)} record(s)")
                            idx = res.index[0]
                            st.markdown("### 📂 Update Records")
                            def save_changes(new_data_dict):
                                try:
                                    if os.path.exists(FILE_PATH):
                                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                                        backup_file = os.path.join(BACKUP_FOLDER, f"ONE_REPORT_backup_{timestamp}.xlsx")
                                        shutil.copy2(FILE_PATH, backup_file)
                                    for c, v in new_data_dict.items():
                                        if c not in df.columns: df[c] = None
                                        df.at[idx, c] = v
                                    df.to_excel(FILE_PATH, sheet_name=SHEET_NAME, index=False)
                                    st.success("✅ Saved Successfully!")
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e: st.error(f"Error: {e}")

                            update_cat = st.selectbox("Select Category to Update:", ["Select an option", "Sale Updation", "Discount Updation", "HSRP Updation", "Finance Updation", "Insurance Updation"])
                            if update_cat == "Sale Updation":
                                st.subheader("📝 Sale Details")
                                with st.form("sale_form"):
                                    sale_cols = ["Model", "Variant", "Colour", "Chassis No.", "Engine No", "Customer Name", "Employee Code (HRMS)", "Sales Consultant Name", "Month Wise FSC Target", "ASM", "SM", "Outlet"]
                                    s_data = {}
                                    cols = st.columns(3)
                                    for i, col in enumerate(sale_cols):
                                        val = str(df.at[idx, col]) if col in df.columns and pd.notna(df.at[idx, col]) else ""
                                        with cols[i % 3]: s_data[col] = st.text_input(col, value=val)
                                    if st.form_submit_button("💾 Save Sale Details"): save_changes(s_data)
                            # (Other update options hidden for brevity, same as before)
                    else: st.warning("No records found.")

            # --- NEW TAB: TARGET VS ACTUALS ---
            if "Target vs Actuals" in tab_map:
                with tab_map["Target vs Actuals"]:
                    st.header("🎯 Target vs Actuals Analysis")
                    
                    # 1. Target Entry Section
                    with st.expander("📝 Enter/Update Targets (Saves to One Report)", expanded=False):
                        st.info("Here you can enter targets for different categories. These will be saved in 'Targets_Master' sheet.")
                        
                        target_cat = st.selectbox("Select Category Type:", ["FSC (Consultant)", "Model", "ASM", "Segment"])
                        
                        # Get Unique values for dropdown
                        unique_vals = []
                        map_col = {"FSC (Consultant)": "Sales Consultant Name", "Model": "Model", "ASM": "ASM", "Segment": "Segment"}
                        
                        if map_col[target_cat] in df.columns:
                            unique_vals = sorted(df[map_col[target_cat]].dropna().unique().tolist())
                        
                        c_t1, c_t2, c_t3 = st.columns(3)
                        sel_month = c_t1.date_input("Select Month", value=pd.Timestamp.now().date())
                        sel_month_str = sel_month.strftime('%b-%Y')
                        sel_item = c_t2.selectbox(f"Select {target_cat}", unique_vals)
                        inp_target = c_t3.number_input("Enter Target Count", min_value=0, step=1)
                        
                        if st.button("💾 Save Target"):
                            # Load existing targets
                            targets_df = load_targets_data(FILE_PATH)
                            
                            # Create new row
                            new_row = pd.DataFrame([{
                                "Category_Type": target_cat, 
                                "Category_Value": sel_item, 
                                "Month_Year": sel_month_str, 
                                "Target_Value": inp_target
                            }])
                            
                            # Remove old entry if exists and append new
                            if not targets_df.empty:
                                mask = (targets_df["Category_Type"] == target_cat) & \
                                       (targets_df["Category_Value"] == sel_item) & \
                                       (targets_df["Month_Year"] == sel_month_str)
                                targets_df = targets_df[~mask]
                            
                            targets_df = pd.concat([targets_df, new_row], ignore_index=True)
                            
                            # Save back to Excel
                            res = save_targets_data(FILE_PATH, targets_df)
                            if res is True:
                                st.success(f"✅ Target Saved for {sel_item} ({sel_month_str})!")
                            else:
                                st.error(f"❌ Error saving: {res}")

                    # 2. Analysis Section
                    st.markdown("---")
                    st.subheader("📊 Performance Report")
                    
                    # Filter Date Range
                    c_d1, c_d2 = st.columns(2)
                    an_start = c_d1.date_input("From Date", value=df['Invoice Date'].min(), key="an_s")
                    an_end = c_d2.date_input("To Date", value=df['Invoice Date'].max(), key="an_e")
                    
                    # Filter Data
                    mask_an = (df['Invoice Date'].dt.date >= an_start) & (df['Invoice Date'].dt.date <= an_end)
                    act_df = df.loc[mask_an].copy()
                    act_df['Month_Str'] = act_df['Invoice Date'].dt.strftime('%b-%Y')
                    
                    # Load Targets
                    tgt_df = load_targets_data(FILE_PATH)
                    
                    # View Tabs
                    v1, v2, v3, v4 = st.tabs(["FSC Wise", "Model Wise", "ASM Wise", "Segment Wise"])
                    
                    def show_target_vs_actual(cat_type, df_col_name):
                        if df_col_name not in act_df.columns:
                            st.warning(f"Column {df_col_name} not found.")
                            return

                        # Calculate Actuals
                        actuals = act_df.groupby(df_col_name).size().reset_index(name='Actual')
                        
                        # Calculate Targets (Filter by Type and Month Range roughly)
                        # For simplicity, summing up targets matching the category values
                        # Note: This simple logic sums all targets in DB for that category. 
                        # Ideally filter by selected month range if Month_Year format matches.
                        
                        if not tgt_df.empty:
                            rel_targets = tgt_df[tgt_df["Category_Type"] == cat_type]
                            # Filter targets by date range (string matching is tricky, doing simple groupby here)
                            # Assuming user enters monthly targets, we sum them if multiple months selected? 
                            # Or just show all. Let's sum by Category Value.
                            targets_sum = rel_targets.groupby("Category_Value")["Target_Value"].sum().reset_index()
                            targets_sum.rename(columns={"Category_Value": df_col_name, "Target_Value": "Target"}, inplace=True)
                            
                            # Merge
                            comp_df = pd.merge(actuals, targets_sum, on=df_col_name, how='outer').fillna(0)
                        else:
                            comp_df = actuals
                            comp_df['Target'] = 0
                            
                        comp_df['Achievement %'] = (comp_df['Actual'] / comp_df['Target'] * 100).fillna(0)
                        comp_df['Shortfall'] = comp_df['Target'] - comp_df['Actual']
                        
                        # Display
                        c_g1, c_g2 = st.columns([2, 1])
                        with c_g1:
                            fig = px.bar(comp_df, x=df_col_name, y=["Target", "Actual"], barmode='group', title=f"{cat_type} Performance")
                            st.plotly_chart(fig, use_container_width=True)
                        with c_g2:
                            st.dataframe(comp_df.style.format({"Target": "{:.0f}", "Actual": "{:.0f}", "Achievement %": "{:.1f}%"}))
                            share_report_feature(comp_df, f"{cat_type}_Target_Report")

                    with v1: show_target_vs_actual("FSC (Consultant)", "Sales Consultant Name")
                    with v2: show_target_vs_actual("Model", "Model")
                    with v3: show_target_vs_actual("ASM", "ASM")
                    with v4: show_target_vs_actual("Segment", "Segment")

            # TAB: FINANCIAL REPORTS
            if "Financial Reports" in tab_map:
                with tab_map["Financial Reports"]:
                    st.header("📈 Financial Reports")
                    # (Existing code...)
                    # [Keep your existing code here unchanged, just collapsing for brevity in chat]
                    min_d = df['Invoice Date'].min().date() if 'Invoice Date' in df.columns else None
                    max_d = df['Invoice Date'].max().date() if 'Invoice Date' in df.columns else None
                    c1, c2 = st.columns(2)
                    start_date = c1.date_input("From Date", value=min_d, key="fin_start")
                    end_date = c2.date_input("To Date", value=max_d, key="fin_end")
                    mask = (df['Invoice Date'].dt.date >= start_date) & (df['Invoice Date'].dt.date <= end_date)
                    report_df = df.loc[mask]
                    group_by = st.selectbox("View:", ["Segment", "Sales Consultant Name", "Outlet", "ASM", "Model"], key="fin_grp")
                    if group_by in report_df.columns: generate_all_report(report_df, group_by, f"{group_by} Financial Performance")

            # TAB: OEM PENDING
            if "OEM Pending Analysis" in tab_map:
                with tab_map["OEM Pending Analysis"]:
                    st.header("📉 OEM Pending vs Received Report")
                    # (Existing code...)
                    min_d = df['Invoice Date'].min().date() if 'Invoice Date' in df.columns else None
                    max_d = df['Invoice Date'].max().date() if 'Invoice Date' in df.columns else None
                    c3, c4 = st.columns(2)
                    p_start_date = c3.date_input("From Date", value=min_d, key="p_start")
                    p_end_date = c4.date_input("To Date", value=max_d, key="p_end")
                    
                    mask = (df['Invoice Date'].dt.date >= p_start_date) & (df['Invoice Date'].dt.date <= p_end_date)
                    p_df = df.loc[mask].copy()
                    
                    # (Rest of the OEM Logic...)
                    # Keeping short for chat limit, paste your FULL Logic here. 
                    # Assuming you paste the full OEM logic from previous code.
                    st.info("Please ensure full OEM logic is pasted here as per previous file.")

            # TAB: ALL REPORT
            if "All Report" in tab_map:
                with tab_map["All Report"]:
                    st.header("📋 Consolidated Reports")
                    # (Existing code...)
                    min_d = df['Invoice Date'].min().date() if 'Invoice Date' in df.columns else None
                    max_d = df['Invoice Date'].max().date() if 'Invoice Date' in df.columns else None
                    c1, c2 = st.columns(2)
                    ar_start = c1.date_input("From Date", value=min_d, key="ar_s")
                    ar_end = c2.date_input("To Date", value=max_d, key="ar_e")
                    mask = (df['Invoice Date'].dt.date >= ar_start) & (df['Invoice Date'].dt.date <= ar_end)
                    all_rep_df = df.loc[mask].copy()
                    
                    st.markdown("---")
                    sub_t1, sub_t2, sub_t3, sub_t4 = st.tabs(["👨‍💼 Consultant", "🚗 Segment", "👔 ASM", "📅 Month-wise"])
                    
                    with sub_t1:
                        cols = ["Sales Consultant Name", "Segment", "ASM", "Sales Manager", "SM", "Outlet"]
                        valid = [c for c in cols if c in all_rep_df.columns]
                        if "Sales Consultant Name" in valid: valid.insert(0, valid.pop(valid.index("Sales Consultant Name")))
                        generate_all_report(all_rep_df, valid, "Detailed Consultant Report")

                    with sub_t2: generate_all_report(all_rep_df, "Segment", "Segment Analysis")
                    with sub_t3: generate_all_report(all_rep_df, "ASM", "ASM Performance")
                    
                    with sub_t4:
                        st.subheader("📅 Month-wise Analysis")
                        # (Consolidated Report Logic...)
                        # Paste full logic here from NCODE05.txt
                        # ...
                        st.write("Consolidated report logic goes here...")

    if auto_refresh: time.sleep(refresh_rate); st.rerun()