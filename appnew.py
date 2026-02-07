import pandas as pd
import streamlit as st
import plotly.express as px
import os
import time
import io
import shutil
import numpy as np
import json
import base64 # Added for image handling in HTML

# 1. Page Configuration
st.set_page_config(page_title="Vehicle Sales System", layout="wide")

# --- USER AUTHENTICATION & MANAGEMENT SYSTEM ---
USERS_FILE = "user_db.json"

# Possible Tabs (Modules)
ALL_MODULES = [
    "Dashboard", 
    "Search & Edit", 
    "Financial Reports", 
    "OEM Pending Analysis", 
    "Data Quality Check", 
    "Tally & TOS Reports", 
    "All Report"
]

# Default Users (Created only if file doesn't exist)
DEFAULT_USERS = {
    "admin": {
        "password": "admin123", 
        "role": "admin", 
        "name": "System Admin",
        "access": ALL_MODULES # Admin gets everything
    },
    "manager": {
        "password": "manager1", 
        "role": "manager", 
        "name": "Sales Manager",
        "access": ["Dashboard", "Financial Reports", "OEM Pending Analysis", "Data Quality Check", "All Report"]
    },
    "sales": {
        "password": "sales1", 
        "role": "sales", 
        "name": "Sales Executive",
        "access": ["Dashboard", "OEM Pending Analysis", "All Report"]
    }
}

# --- HELPER: INDIAN CURRENCY FORMATTER ---
def format_lakhs(value):
    """Converts a number to Indian system (Lakhs/Crores) string format."""
    if isinstance(value, (int, float)):
        try:
            val_str = "{:.0f}".format(value)
        except:
            return value
        
        if "." in val_str:
            head, decimal = val_str.split(".")
        else:
            head, decimal = val_str, ""
            
        is_neg = head.startswith("-")
        if is_neg: head = head[1:]
        
        if len(head) <= 3:
            res = head
        else:
            last3 = head[-3:]
            rest = head[:-3]
            # chunks of 2
            rev = rest[::-1]
            chunks = [rev[i:i+2] for i in range(0, len(rev), 2)]
            # join and reverse back
            res = ",".join(chunks)[::-1] + "," + last3
            
        if is_neg: res = "-" + res
        return res + ("." + decimal if decimal else "")
    return value

def load_users():
    """Load users from JSON file, create if missing. AND AUTO-FIX PERMISSIONS."""
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump(DEFAULT_USERS, f)
        return DEFAULT_USERS
    try:
        with open(USERS_FILE, 'r') as f:
            data = json.load(f)
            # Auto-Fix Permissions logic
            if 'admin' in data: data['admin']['access'] = ALL_MODULES
            if 'manager' in data:
                current = data['manager'].get('access', [])
                if "Data Quality Check" not in current: current.insert(3, "Data Quality Check")
                data['manager']['access'] = current
            return data
    except:
        return DEFAULT_USERS

def save_users(users):
    """Save users to JSON file."""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

# Load users at startup and save fixes
users_db = load_users()
save_users(users_db)

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
                if username in current_users and current_users[username]['password'] == password:
                    st.session_state['authenticated'] = True
                    st.session_state['user'] = username
                    st.session_state['role'] = current_users[username]['role']
                    st.session_state['name'] = current_users[username]['name']
                    st.session_state['access'] = current_users[username].get('access', [])
                    st.success(f"Welcome {current_users[username]['name']}!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Invalid Credentials")

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# --- MAIN APP ---
if not st.session_state['authenticated']:
    login_page()
else:
    # --- SIDEBAR: USER INFO & SETTINGS ---
    with st.sidebar:
        st.info(f"👤 User: **{st.session_state['name']}**\n🔑 Role: **{st.session_state['role'].upper()}**")
        
        with st.expander("🔑 Change My Password"):
            curr_pass = st.text_input("Current Password", type="password", key="cp")
            new_pass = st.text_input("New Password", type="password", key="np")
            conf_pass = st.text_input("Confirm New Password", type="password", key="cnp")
            if st.button("Update Password"):
                users = load_users()
                uname = st.session_state['user']
                if users[uname]['password'] == curr_pass:
                    if new_pass == conf_pass and new_pass:
                        users[uname]['password'] = new_pass
                        save_users(users)
                        st.success("✅ Password Changed!")
                    else: st.error("❌ New passwords do not match or empty.")
                else: st.error("❌ Incorrect Current Password.")

        if st.session_state['role'] == 'admin':
            with st.expander("🛠️ Admin: User Management", expanded=False):
                st.markdown("##### ➕ Create / Update User")
                u_new = st.text_input("Username (Unique)")
                p_new = st.text_input("Set Password", type="password")
                n_new = st.text_input("Display Name")
                r_new = st.selectbox("Assign Role Label", ["admin", "manager", "sales"])
                a_new = st.multiselect("Allowed Tabs", ALL_MODULES, default=ALL_MODULES if r_new == 'admin' else ["Dashboard"])
                if st.button("Create/Update User"):
                    if u_new and p_new and n_new and a_new:
                        users = load_users()
                        users[u_new] = {"password": p_new, "role": r_new, "name": n_new, "access": a_new}
                        save_users(users)
                        st.success(f"✅ User '{u_new}' Saved!")
                    else: st.error("All fields required.")
                
                st.markdown("---")
                st.markdown("##### 🗑️ Delete User")
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

        st.markdown("---")
        if st.button("🚪 Logout", type="primary"):
            st.session_state['authenticated'] = False
            st.rerun()

    # --- MAIN APPLICATION LOGIC ---
    FILE_PATH = "ONE REPORT.xlsx"
    SHEET_NAME = "Retail Format"
    DB_FOLDER = "tally_tos_database"
    if not os.path.exists(DB_FOLDER): os.makedirs(DB_FOLDER)
    FILES_DB = {
        "Tally Sale": os.path.join(DB_FOLDER, "master_tally_sale.csv"),
        "Tally Purchase": os.path.join(DB_FOLDER, "master_tally_purchase.csv"),
        "TOS In": os.path.join(DB_FOLDER, "master_tos_in.csv"),
        "TOS Out": os.path.join(DB_FOLDER, "master_tos_out.csv")
    }

    st.sidebar.title("⚙️ Settings")
    auto_refresh = st.sidebar.checkbox("✅ Enable Auto-Update", value=True)
    refresh_rate = st.sidebar.slider("Refresh Rate (s)", 5, 60, 10)

    def get_file_timestamp(): return os.path.getmtime(FILE_PATH) if os.path.exists(FILE_PATH) else 0

    @st.cache_data
    def load_data(last_modified_time):
        if not os.path.exists(FILE_PATH): return None
        try:
            df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
            df.columns = df.columns.str.strip()
            if 'Invoice Date' in df.columns: df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], dayfirst=True, errors='coerce')
            if 'Chassis No.' in df.columns: df['Chassis No.'] = df['Chassis No.'].astype(str)
            target_cols = ['Sale Invoice Amount With GST', 'Sale Invoice Amount Basic Value', 'Purchase With GST Value', 'Purchase Basic Value', 'TOTAL OEM DISCOUNTS', 'TOTAL INTENAL DISCOUNTS', 'TOTAL OEM & INTERNAL NET DISCOUNTS', 'TOTAL Credit Note NET DISCOUNT', 'MARGIN', 'TOTAL RECEIVED OEM NET DISCOUNTS', 'FINAL MARGIN', 'OEM - RETAIL SCHEME', 'RECEIVED OEM - RETAIL SCHEME', 'OEM - CORPORATE SCHEME', 'RECEIVED OEM - CORPORATE SCHEME', 'OEM - EXCHANGE SCHEME', 'RECEIVED OEM - EXCHANGE SCHEME', 'OEM - SPECIAL SCHEME', 'RECEIVED OEM - SPECIAL SCHEME', 'OEM - WHOLESALE SUPPORT', 'RECEIVED OEM - WHOLESALE SUPPORT', 'OEM - LOYALTY BONUS', 'RECEIVED OEM - LOYALTY BONUS', 'OEM - OTHERS', 'RECEIVED OEM - OTHERS', 'TOTAL Credit Note Amout OEM', 'INTERNAL - RETAIL SCHEME', 'INTERNAL - CORPORATE SCHEME', 'INTERNAL - EXCHANGE SUPPORT', 'INTERNAL - Accesories Discount', 'INTERNAL - Dealer Cash Discount', 'INTERNAL - Employee Discount', 'INTERNAL - Referal Bonus', 'INTERNAL - EW Scheme', 'INTERNAL - Depreciation', 'INTERNAL - Other discounts', 'INTERNAL - Additional Special discount', 'INTERNAL - Loyalty Scheme']
            for col in target_cols:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                else: df[col] = 0
            return df
        except Exception as e: st.error(f"Error: {e}"); return None

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
        
        # --- FIX: Handle MultiIndex vs Single Index for Grand Total ---
        if len(group_cols) > 1:
            # Create a tuple for MultiIndex assignment
            gt_name = tuple(['GRAND TOTAL'] + [''] * (len(group_cols) - 1))
        else:
            # Simple string for Single Index
            gt_name = 'GRAND TOTAL'
            
        pivot_data.loc[gt_name, :] = gt_row
        pivot_data.columns = [c.strftime('%b-%Y') if isinstance(c, pd.Period) else c for c in pivot_data.columns]
        return pivot_data

    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f: data = f.read()
        return base64.b64encode(data).decode()

    # --- HEADER ---
    logo_left_path = "logo_left.png"
    logo_right_path = "logo_right.png"
    logo_left_html = f'<img src="data:image/png;base64,{get_base64_of_bin_file(logo_left_path)}" style="height: 80px;">' if os.path.exists(logo_left_path) else ""
    logo_right_html = f'<img src="data:image/png;base64,{get_base64_of_bin_file(logo_right_path)}" style="height: 80px;">' if os.path.exists(logo_right_path) else ""
    st.markdown(f"""<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0;">
        <div style="flex: 1; text-align: left;">{logo_left_html}</div>
        <div style="flex: 6; text-align: center;"><h1 style="margin: 0;">🚗 Vehicle Sales Management System</h1></div>
        <div style="flex: 1; text-align: right;">{logo_right_html}</div>
    </div><hr style="margin-top: 0; margin-bottom: 20px;">""", unsafe_allow_html=True)

    ts = get_file_timestamp()
    if ts == 0: st.error("❌ File not found!")
    else:
        df = load_data(ts)
        if df is not None:
            allowed_tabs = st.session_state.get('access', [])
            if not allowed_tabs: allowed_tabs = ["Dashboard"]
            tabs = st.tabs(allowed_tabs)
            tab_map = {name: tab for name, tab in zip(allowed_tabs, tabs)}

            # --- MODULES ---
            if "Dashboard" in tab_map:
                with tab_map["Dashboard"]:
                    st.subheader("Overview")
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Total Vehicles", format_lakhs(len(df)))
                    k2.metric("Total Revenue", f"₹ {format_lakhs(df['Sale Invoice Amount With GST'].sum())}")
                    k3.metric("Total Final Margin", f"₹ {format_lakhs(df['FINAL MARGIN'].sum())}")
                    st.dataframe(df)

            if "Search & Edit" in tab_map:
                with tab_map["Search & Edit"]:
                    st.header("Search & Edit Records")
                    search_mode = st.selectbox("Search By:", ["General Search", "Chassis No. (Last 8 / Full)"])
                    search_val = st.text_input("Enter Value to Search").strip()
                    if search_val:
                        if search_mode == "Chassis No. (Last 8 / Full)": mask = df['Chassis No.'].str.contains(search_val, case=False, na=False) if 'Chassis No.' in df.columns else pd.Series([False]*len(df))
                        else: mask = df.apply(lambda x: x.astype(str).str.contains(search_val, case=False, na=False)).any(axis=1)
                        res = df[mask]
                        if not res.empty:
                            st.success(f"Found {len(res)} record(s)")
                            idx = res.index[0]
                            st.markdown("### 📂 Update Records")
                            def save_changes(new_data_dict):
                                try:
                                    for c, v in new_data_dict.items():
                                        if c not in df.columns: df[c] = None
                                        df.at[idx, c] = v
                                    df.to_excel(FILE_PATH, sheet_name=SHEET_NAME, index=False)
                                    st.success("✅ Saved Successfully!"); time.sleep(1); st.rerun()
                                except Exception as e: st.error(f"Error: {e}")
                            
                            update_cat = st.selectbox("Select Category to Update:", ["Select", "Sale Updation", "Discount Updation", "HSRP Updation", "Finance Updation", "Insurance Updation"])
                            if update_cat == "Sale Updation":
                                with st.form("sale_form"):
                                    sale_cols = ["Model", "Variant", "Colour", "Chassis No.", "Engine No", "Customer Name", "Employee Code (HRMS)", "Sales Consultant Name", "Month Wise FSC Target", "ASM", "SM", "Outlet"]
                                    s_data = {}
                                    cols = st.columns(3)
                                    for i, col in enumerate(sale_cols):
                                        val = str(df.at[idx, col]) if col in df.columns and pd.notna(df.at[idx, col]) else ""
                                        with cols[i % 3]: s_data[col] = st.text_input(col, value=val)
                                    if st.form_submit_button("💾 Save"): save_changes(s_data)

            if "Financial Reports" in tab_map:
                with tab_map["Financial Reports"]:
                    st.header("📈 Financial Reports")
                    c1, c2 = st.columns(2)
                    min_d = df['Invoice Date'].min().date() if 'Invoice Date' in df.columns else None
                    max_d = df['Invoice Date'].max().date() if 'Invoice Date' in df.columns else None
                    start_date = c1.date_input("From Date", value=min_d, key="fin_start")
                    end_date = c2.date_input("To Date", value=max_d, key="fin_end")
                    mask = (df['Invoice Date'].dt.date >= start_date) & (df['Invoice Date'].dt.date <= end_date)
                    report_df = df.loc[mask]
                    group_by = st.selectbox("View:", ["Segment", "Sales Consultant Name", "Outlet", "ASM", "Model"], key="fin_grp")
                    if group_by in report_df.columns: generate_all_report(report_df, group_by, f"{group_by} Financial Performance")

            if "OEM Pending Analysis" in tab_map:
                with tab_map["OEM Pending Analysis"]:
                    st.header("📉 OEM Pending vs Received Report")
                    c3, c4 = st.columns(2)
                    p_start_date = c3.date_input("From Date", value=min_d, key="p_start")
                    p_end_date = c4.date_input("To Date", value=max_d, key="p_end")
                    mask = (df['Invoice Date'].dt.date >= p_start_date) & (df['Invoice Date'].dt.date <= p_end_date)
                    p_df = df.loc[mask].copy()
                    
                    scheme_pairs = [('OEM - RETAIL SCHEME', 'RECEIVED OEM - RETAIL SCHEME', 'Pending Retail'), ('OEM - CORPORATE SCHEME', 'RECEIVED OEM - CORPORATE SCHEME', 'Pending Corporate'), ('OEM - EXCHANGE SCHEME', 'RECEIVED OEM - EXCHANGE SCHEME', 'Pending Exchange'), ('OEM - SPECIAL SCHEME', 'RECEIVED OEM - SPECIAL SCHEME', 'Pending Special'), ('OEM - WHOLESALE SUPPORT', 'RECEIVED OEM - WHOLESALE SUPPORT', 'Pending Wholesale'), ('OEM - LOYALTY BONUS', 'RECEIVED OEM - LOYALTY BONUS', 'Pending Loyalty'), ('OEM - OTHERS', 'RECEIVED OEM - OTHERS', 'Pending Others')]
                    
                    p_df['PENDING_TOTAL'] = p_df['TOTAL OEM DISCOUNTS'] - p_df['TOTAL RECEIVED OEM NET DISCOUNTS']
                    p_df['STATUS'] = p_df['PENDING_TOTAL'].apply(lambda x: "PENDING" if x > 1 else "RECEIVED/CLEARED")
                    
                    # KPIs
                    tot_pend = p_df[p_df['STATUS']=="PENDING"]['PENDING_TOTAL'].sum()
                    tot_rec = p_df['TOTAL RECEIVED OEM NET DISCOUNTS'].sum()
                    rec_rate = (tot_rec / p_df['TOTAL OEM DISCOUNTS'].sum() * 100) if p_df['TOTAL OEM DISCOUNTS'].sum() > 0 else 0
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Total Pending", f"₹ {format_lakhs(tot_pend)}"); k2.metric("Total Received", f"₹ {format_lakhs(tot_rec)}"); k3.metric("Recovery Rate", f"{rec_rate:.1f}%")
                    st.markdown("---")

                    base_cols = ['Chassis No.', 'Customer Name', 'Invoice No.', 'Invoice Date', 'Model', 'Outlet', 'Sales Consultant Name']
                    valid_base = [c for c in base_cols if c in p_df.columns]
                    
                    # Pending List
                    pending_export_df = p_df[p_df['STATUS'] == "PENDING"].copy()
                    if not pending_export_df.empty:
                        for given, received, pending_name in scheme_pairs:
                            if given in pending_export_df.columns and received in pending_export_df.columns:
                                pending_export_df[pending_name] = pending_export_df[given] - pending_export_df[received]
                        st.download_button("Download PENDING List", data=to_excel(pending_export_df[valid_base + [p[2] for p in scheme_pairs] + ['PENDING_TOTAL']]), file_name="Detailed_Pending_List.xlsx")
                    
                    # Scheme Wise Summary (NEW ENHANCED)
                    st.markdown("---")
                    st.subheader("📑 Scheme Wise Performance")
                    summary_data = []
                    for given, received, pending_name in scheme_pairs:
                        if given in p_df.columns and received in p_df.columns:
                            s_given = p_df[given].sum(); s_rec = p_df[received].sum(); s_pend = s_given - s_rec
                            status = "Shortage" if s_pend > 0 else ("Excess" if s_pend < 0 else "Balanced")
                            s_rec_pct = (s_rec / s_given * 100) if s_given > 0 else 0
                            summary_data.append({"Scheme Type": pending_name.replace("Pending ", ""), "Total OEM Discounts": s_given, "Actual OEM Received": s_rec, "Pending/Excess Amount": s_pend, "Status": status, "Recovery %": s_rec_pct})
                    
                    if summary_data:
                        summ_df = pd.DataFrame(summary_data)
                        gt_g = summ_df["Total OEM Discounts"].sum(); gt_r = summ_df["Actual OEM Received"].sum(); gt_p = summ_df["Pending/Excess Amount"].sum()
                        gt_row = pd.DataFrame([{"Scheme Type": "GRAND TOTAL", "Total OEM Discounts": gt_g, "Actual OEM Received": gt_r, "Pending/Excess Amount": gt_p, "Status": "-", "Recovery %": (gt_r/gt_g*100) if gt_g>0 else 0}])
                        summ_df = pd.concat([summ_df, gt_row], ignore_index=True)
                        
                        def highlight_status(val): return 'color: red; font-weight: bold' if val == 'Shortage' else ('color: green; font-weight: bold' if val == 'Excess' else '')
                        st.dataframe(summ_df.style.format({"Total OEM Discounts": lambda x: f"₹ {format_lakhs(x)}", "Actual OEM Received": lambda x: f"₹ {format_lakhs(x)}", "Pending/Excess Amount": lambda x: f"₹ {format_lakhs(x)}", "Recovery %": "{:.1f}%"}).applymap(highlight_status, subset=['Status']))

                        # Detailed Download
                        det_df = p_df[valid_base].copy()
                        color_cols_p = []; color_cols_r = []; color_cols_g = []
                        for given, received, pending_name in scheme_pairs:
                            if given in p_df.columns and received in p_df.columns:
                                s = pending_name.replace("Pending ", "")
                                det_df[f"{s} - Given"] = p_df[given]; color_cols_g.append(f"{s} - Given")
                                det_df[f"{s} - Received"] = p_df[received]; color_cols_r.append(f"{s} - Received")
                                det_df[f"{s} - Pending"] = p_df[given] - p_df[received]; color_cols_p.append(f"{s} - Pending")
                        
                        det_df["TOTAL DIFFERENCE"] = p_df["PENDING_TOTAL"]
                        
                        def to_styled_excel(df):
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                styler = df.style
                                styler.applymap(lambda v: 'color: red; font-weight: bold' if isinstance(v, (int, float)) and v > 0 else '', subset=color_cols_p + ["TOTAL DIFFERENCE"])
                                styler.to_excel(writer, index=False, sheet_name='Scheme_Wise')
                            return output.getvalue()
                        st.download_button("Download Detailed Scheme Wise Report (Color Coded)", data=to_styled_excel(det_df), file_name="Detailed_Scheme_Wise_Report.xlsx")

                    # Credit Note
                    st.markdown("---")
                    st.subheader("💳 OEM Credit Note Analysis")
                    if 'TOTAL Credit Note Amout OEM' in p_df.columns:
                        cn_df = p_df[p_df['TOTAL Credit Note Amout OEM'] > 0]
                        if not cn_df.empty:
                            st.dataframe(cn_df[valid_base + ['TOTAL Credit Note Amout OEM', 'Credit Note Reference No', 'RECEIVED OEM REMARKS (IF ANY REASON OR CREDIT NOTE NO)']])
                            st.download_button("Download Credit Notes", data=to_excel(cn_df), file_name="Credit_Notes.xlsx")
                        else: st.info("No Credit Notes found.")

            # TAB: DATA QUALITY CHECK (DEEP LOGIC)
            if "Data Quality Check" in tab_map:
                with tab_map["Data Quality Check"]:
                    st.header("🛡️ Data Quality Inspector (Deep Scan)")
                    st.info("Scans for: Duplicates, Missing Data, Negative Values, Totals Mismatch, Mobile Numbers, Future Dates, Zero Sales.")
                    
                    if st.button("Run Deep Quality Check", type="primary"):
                        errors = []
                        
                        # 1. Duplicates
                        if 'Chassis No.' in df.columns:
                            for i, row in df[df.duplicated('Chassis No.', keep=False)].iterrows():
                                errors.append({"Row": i+2, "Chassis": row['Chassis No.'], "Customer": row.get('Customer Name', ''), "Issue": "Duplicate Chassis", "Value": row['Chassis No.']})
                        
                        # 2. Missing Data
                        for col in ['Invoice Date', 'Customer Name', 'Model', 'Outlet']:
                            if col in df.columns:
                                for i, row in df[df[col].isna() | (df[col] == '')].iterrows():
                                    errors.append({"Row": i+2, "Chassis": row.get('Chassis No.', ''), "Customer": row.get('Customer Name', ''), "Issue": "Missing Data", "Value": f"Column '{col}' is empty"})

                        # 3. Negatives
                        money_cols = [c for c in df.columns if any(x in c.upper() for x in ['AMOUNT', 'PRICE', 'VALUE', 'MARGIN', 'DISCOUNT'])]
                        for col in money_cols:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                for i, row in df[df[col] < 0].iterrows():
                                    errors.append({"Row": i+2, "Chassis": row.get('Chassis No.', ''), "Customer": row.get('Customer Name', ''), "Issue": "Negative Value", "Value": f"{col}: {row[col]}"})

                        # 4a. OEM Discount Sum Check (Given)
                        oem_components = ['OEM - RETAIL SCHEME', 'OEM - CORPORATE SCHEME', 'OEM - EXCHANGE SCHEME', 'OEM - SPECIAL SCHEME', 'OEM - WHOLESALE SUPPORT', 'OEM - LOYALTY BONUS', 'OEM - OTHERS']
                        if 'TOTAL OEM DISCOUNTS' in df.columns:
                            valid_comps = [c for c in oem_components if c in df.columns]
                            if valid_comps:
                                for c in valid_comps: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
                                df['TOTAL OEM DISCOUNTS'] = pd.to_numeric(df['TOTAL OEM DISCOUNTS'], errors='coerce').fillna(0)
                                calc_oem = df[valid_comps].sum(axis=1)
                                mismatch = df[abs(calc_oem - df['TOTAL OEM DISCOUNTS']) > 1]
                                for i, row in mismatch.iterrows():
                                     errors.append({"Row": i+2, "Chassis": row.get('Chassis No.', ''), "Customer": row.get('Customer Name', ''), "Issue": "OEM Given Mismatch", "Value": f"Sum: {calc_oem[i]} vs Total: {row['TOTAL OEM DISCOUNTS']}"})

                        # 4b. OEM Discount Sum Check (Received) - NEW REQUEST
                        rec_oem_components = ['RECEIVED OEM - RETAIL SCHEME', 'RECEIVED OEM - CORPORATE SCHEME', 'RECEIVED OEM - EXCHANGE SCHEME', 'RECEIVED OEM - SPECIAL SCHEME', 'RECEIVED OEM - WHOLESALE SUPPORT', 'RECEIVED OEM - LOYALTY BONUS', 'RECEIVED OEM - OTHERS']
                        if 'TOTAL RECEIVED OEM NET DISCOUNTS' in df.columns:
                            valid_rec_comps = [c for c in rec_oem_components if c in df.columns]
                            if valid_rec_comps:
                                for c in valid_rec_comps: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
                                df['TOTAL RECEIVED OEM NET DISCOUNTS'] = pd.to_numeric(df['TOTAL RECEIVED OEM NET DISCOUNTS'], errors='coerce').fillna(0)
                                calc_rec_oem = df[valid_rec_comps].sum(axis=1)
                                mismatch_rec = df[abs(calc_rec_oem - df['TOTAL RECEIVED OEM NET DISCOUNTS']) > 1]
                                for i, row in mismatch_rec.iterrows():
                                     errors.append({"Row": i+2, "Chassis": row.get('Chassis No.', ''), "Customer": row.get('Customer Name', ''), "Issue": "OEM Received Mismatch", "Value": f"Sum: {calc_rec_oem[i]} vs Total: {row['TOTAL RECEIVED OEM NET DISCOUNTS']}"})

                        # 5. Internal Discount Sum Check
                        int_components = ['INTERNAL - RETAIL SCHEME', 'INTERNAL - CORPORATE SCHEME', 'INTERNAL - EXCHANGE SUPPORT', 'INTERNAL - Accesories Discount', 'INTERNAL - Dealer Cash Discount', 'INTERNAL - Employee Discount', 'INTERNAL - Referal Bonus', 'INTERNAL - EW Scheme', 'INTERNAL - Depreciation', 'INTERNAL - Other discounts', 'INTERNAL - Additional Special discount', 'INTERNAL - Loyalty Scheme']
                        if 'TOTAL INTENAL DISCOUNTS' in df.columns: 
                             valid_int_comps = [c for c in int_components if c in df.columns]
                             if valid_int_comps:
                                 df['Calc_Int'] = df[valid_int_comps].sum(axis=1)
                                 mismatch_int = df[abs(df['Calc_Int'] - df['TOTAL INTENAL DISCOUNTS']) > 1]
                                 for i, row in mismatch_int.iterrows():
                                     errors.append({"Row": i+2, "Chassis": row.get('Chassis No.', ''), "Customer": row.get('Customer Name', ''), "Issue": "Internal Total Mismatch", "Value": f"Sum: {row['Calc_Int']} vs Total: {row['TOTAL INTENAL DISCOUNTS']}"})

                        # 6. Mobile Number
                        if 'Customer Mobile No.' in df.columns:
                            mobiles = df['Customer Mobile No.'].astype(str).str.replace(r'\.0$', '', regex=True)
                            invalid = mobiles[mobiles.apply(lambda x: len(x) != 10 and x.lower() != 'nan' and x != '')]
                            for i in invalid.index:
                                errors.append({"Row": i+2, "Chassis": df.at[i, 'Chassis No.'], "Customer": df.at[i, 'Customer Name'], "Issue": "Invalid Mobile", "Value": mobiles[i]})

                        if errors:
                            err_df = pd.DataFrame(errors)
                            st.error(f"Found {len(errors)} Data Quality Issues!")
                            st.dataframe(err_df)
                            st.download_button("Download Error Report", data=to_excel(err_df), file_name="Data_Quality_Errors.xlsx")
                        else: st.success("✅ Clean Data! No logical or format errors found.")

            # TAB: TALLY & TOS
            if "Tally & TOS Reports" in tab_map:
                with tab_map["Tally & TOS Reports"]:
                    st.header("📑 Tally & TOS")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.file_uploader("Upload Tally Sale", key="u1"): save_and_append(st.session_state.u1, FILES_DB["Tally Sale"])
                        if st.file_uploader("Upload Tally Purchase", key="u2"): save_and_append(st.session_state.u2, FILES_DB["Tally Purchase"])
                    with c2:
                        if st.file_uploader("Upload TOS In", key="u3"): save_and_append(st.session_state.u3, FILES_DB["TOS In"])
                        if st.file_uploader("Upload TOS Out", key="u4"): save_and_append(st.session_state.u4, FILES_DB["TOS Out"])
                    if st.button("Clean Duplicates"): remove_duplicates_only(FILES_DB["Tally Sale"]); st.success("Cleaned!")

            # TAB: ALL REPORT
            if "All Report" in tab_map:
                with tab_map["All Report"]:
                    st.header("📋 Consolidated Reports")
                    c1, c2 = st.columns(2)
                    ar_start = c1.date_input("From Date", value=min_d, key="ar_s")
                    ar_end = c2.date_input("To Date", value=max_d, key="ar_e")
                    mask = (df['Invoice Date'].dt.date >= ar_start) & (df['Invoice Date'].dt.date <= ar_end)
                    all_rep_df = df.loc[mask].copy()

                    # Report Selection Dropdown
                    report_type = st.selectbox("Select Report:", [
                        "1. Consultant & Segment", "2. ASM Performance", "3. Model Wise Performance", 
                        "4. Consultant Wise Sale", "5. Consultant Consolidate", "6. Consultant Consolidate (Model Wise)"
                    ])
                    
                    st.markdown("---")
                    
                    if report_type == "1. Consultant & Segment":
                        grp = [c for c in ["Sales Consultant Name", "ASM", "Sales Manager", "Segment"] if c in all_rep_df.columns]
                        if grp: st.dataframe(generate_month_wise_pivot(all_rep_df, grp, start_date=ar_start, end_date=ar_end).style.format(format_lakhs).format(subset=["Average"], formatter="{:.1f}"))

                    elif report_type == "2. ASM Performance":
                        grp = [c for c in ["ASM", "Segment"] if c in all_rep_df.columns]
                        if grp: st.dataframe(generate_month_wise_pivot(all_rep_df, grp, start_date=ar_start, end_date=ar_end).style.format(format_lakhs).format(subset=["Average"], formatter="{:.1f}"))

                    elif report_type == "3. Model Wise Performance":
                        grp = [c for c in ["Model", "Segment"] if c in all_rep_df.columns]
                        if grp: st.dataframe(generate_month_wise_pivot(all_rep_df, grp, start_date=ar_start, end_date=ar_end).style.format(format_lakhs).format(subset=["Average"], formatter="{:.1f}"))
                    
                    elif report_type == "4. Consultant Wise Sale":
                        if "Sales Consultant Name" in all_rep_df.columns:
                            st.dataframe(generate_month_wise_pivot(all_rep_df, ["Sales Consultant Name"], start_date=ar_start, end_date=ar_end).style.format(format_lakhs).format(subset=["Average"], formatter="{:.1f}"))

                    # Helper for Consolidate Logic
                    def create_consolidate(grp_cols):
                         if not grp_cols: return pd.DataFrame()
                         df_c = all_rep_df.copy()
                         for c in grp_cols: df_c[c] = df_c[c].fillna("Unknown")
                         
                         # Finance Logic
                         if "FINANCE IN/OUT" in df_c.columns:
                             df_c['Fin_Type'] = df_c["FINANCE IN/OUT"].astype(str).str.strip().str.upper()
                             df_c['Fin_Name'] = df_c.get('Name of the Financier', '').astype(str).str.strip().str.upper()
                             df_c['Other Fin In-House'] = ((df_c['Fin_Type']=='IN HOUSE') & (~df_c['Fin_Name'].str.contains('MAHINDRA', case=False))).astype(int)
                             df_c['MMFSL Fin In-House'] = ((df_c['Fin_Type']=='IN HOUSE') & (df_c['Fin_Name'].str.contains('MAHINDRA', case=False))).astype(int)
                             df_c['Total IN HOUSE'] = (df_c['Fin_Type']=='IN HOUSE').astype(int)
                             df_c['OWN FUNDS / LEASING'] = df_c['Fin_Type'].isin(['OWN FUNDS', 'LEASE']).astype(int)
                             df_c['Total OUT HOUSE'] = (df_c['Fin_Type']=='OUT HOUSE').astype(int)
                             df_c['Total Fin (In+Out)'] = (df_c['Fin_Type'].isin(['IN HOUSE', 'OUT HOUSE'])).astype(int)
                         
                         # Insurance Logic
                         if "INSURANCE IN/OUT" in df_c.columns:
                             df_c['Ins_Type'] = df_c["INSURANCE IN/OUT"].astype(str).str.strip().str.upper()
                             df_c['Total In-House (Ins)'] = (df_c['Ins_Type']=='IN HOUSE').astype(int)
                             df_c['Total Out-House (Ins)'] = (df_c['Ins_Type']=='OUT HOUSE').astype(int)
                         
                         df_c['Tally Sales Count'] = 1
                         
                         agg_cols = ['Other Fin In-House', 'MMFSL Fin In-House', 'Total IN HOUSE', 'OWN FUNDS / LEASING', 'Total OUT HOUSE', 'Total Fin (In+Out)', 'Tally Sales Count', 'Total In-House (Ins)', 'Total Out-House (Ins)']
                         valid_agg = [c for c in agg_cols if c in df_c.columns]
                         
                         grouped = df_c.groupby(grp_cols)[valid_agg].sum()
                         
                         # Calculated Columns
                         if 'Total IN HOUSE' in grouped.columns:
                             grouped['Finance Grand Total'] = grouped['Total IN HOUSE'] + grouped.get('OWN FUNDS / LEASING', 0) + grouped.get('Total OUT HOUSE', 0)
                             grouped['Difference'] = grouped['Tally Sales Count'] - grouped['Finance Grand Total']
                             if 'Total Fin (In+Out)' in grouped.columns:
                                 grouped['Fin In-House %'] = (grouped['Total IN HOUSE'] / grouped['Total Fin (In+Out)'] * 100).fillna(0)
                             if 'MMFSL Fin In-House' in grouped.columns:
                                 grouped['MMFSL Share %'] = (grouped['MMFSL Fin In-House'] / grouped['Total IN HOUSE'] * 100).fillna(0)

                         if 'Total In-House (Ins)' in grouped.columns:
                             grouped['Insurance Grand Total'] = grouped['Total In-House (Ins)'] + grouped.get('Total Out-House (Ins)', 0)
                             grouped['Ins In-House %'] = (grouped['Total In-House (Ins)'] / grouped['Tally Sales Count'] * 100).fillna(0)
                         
                         # Grand Total
                         gt = grouped.sum()
                         # Re-calc percentages for GT
                         if 'Total Fin (In+Out)' in gt and gt['Total Fin (In+Out)'] > 0: gt['Fin In-House %'] = gt['Total IN HOUSE']/gt['Total Fin (In+Out)']*100
                         if 'Total IN HOUSE' in gt and gt['Total IN HOUSE'] > 0: gt['MMFSL Share %'] = gt['MMFSL Fin In-House']/gt['Total IN HOUSE']*100
                         if 'Tally Sales Count' in gt and gt['Tally Sales Count'] > 0: gt['Ins In-House %'] = gt['Total In-House (Ins)']/gt['Tally Sales Count']*100
                         
                         # Handle MultiIndex for Grand Total Row assignment
                         if len(grp_cols) > 1:
                             gt_name = tuple(['GRAND TOTAL'] + [''] * (len(grp_cols) - 1))
                         else:
                             gt_name = 'GRAND TOTAL'
                         
                         grouped.loc[gt_name, :] = gt
                         
                         # Merge Monthly
                         piv = generate_month_wise_pivot(all_rep_df, grp_cols, start_date=ar_start, end_date=ar_end)
                         piv = piv.drop(columns=['Total', 'Average'], errors='ignore')
                         
                         final = pd.merge(grouped.iloc[:-1], piv, left_index=True, right_index=True, how='left')
                         
                         # Add GT row back manually
                         gt_row_combined = pd.concat([grouped.loc[[gt_name]], piv.loc[[gt_name]]], axis=1)
                         final = pd.concat([final, gt_row_combined])
                         
                         return final

                    if report_type == "5. Consultant Consolidate":
                         grp = [c for c in ["Sales Consultant Name", "Segment", "ASM", "Sales Manager", "SM", "Outlet"] if c in all_rep_df.columns]
                         if grp: st.dataframe(create_consolidate(grp).style.format(format_lakhs).format(subset=['Fin In-House %', 'MMFSL Share %', 'Ins In-House %'], formatter="{:.1f}"))

                    elif report_type == "6. Consultant Consolidate (Model Wise)":
                         grp = [c for c in ["Sales Consultant Name", "Segment", "Model", "ASM", "Sales Manager", "SM", "Outlet"] if c in all_rep_df.columns]
                         if grp: st.dataframe(create_consolidate(grp).style.format(format_lakhs).format(subset=['Fin In-House %', 'MMFSL Share %', 'Ins In-House %'], formatter="{:.1f}"))

    if auto_refresh: time.sleep(refresh_rate); st.rerun()