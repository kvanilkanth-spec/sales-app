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
    "Data Quality Check", # ADDED BACK
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
    """Load users from JSON file, create if missing."""
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump(DEFAULT_USERS, f)
        return DEFAULT_USERS
    try:
        with open(USERS_FILE, 'r') as f:
            data = json.load(f)
            # Compatibility fix: If old DB doesn't have 'access' field, add default
            for u in data:
                if 'access' not in data[u]:
                    if data[u]['role'] == 'admin': data[u]['access'] = ALL_MODULES
                    elif data[u]['role'] == 'manager': data[u]['access'] = ["Dashboard", "Financial Reports", "OEM Pending Analysis", "Data Quality Check", "All Report"]
                    else: data[u]['access'] = ["Dashboard", "OEM Pending Analysis", "All Report"]
            return data
    except:
        return DEFAULT_USERS

def save_users(users):
    """Save users to JSON file."""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

# Load users at startup
users_db = load_users()

def login_page():
    st.markdown("<h1 style='text-align: center;'>🔒 Secure Login</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                # Refresh DB to ensure latest data
                current_users = load_users()
                if username in current_users and current_users[username]['password'] == password:
                    st.session_state['authenticated'] = True
                    st.session_state['user'] = username
                    st.session_state['role'] = current_users[username]['role']
                    st.session_state['name'] = current_users[username]['name']
                    # LOAD ACCESS RIGHTS
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
        
        # 1. CHANGE OWN PASSWORD (For All Users)
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
                    else:
                        st.error("❌ New passwords do not match or empty.")
                else:
                    st.error("❌ Incorrect Current Password.")

        # 2. ADMIN PANEL: CREATE/MANAGE USERS (UPDATED WITH ACCESS CONTROL)
        if st.session_state['role'] == 'admin':
            with st.expander("🛠️ Admin: User Management", expanded=False):
                st.markdown("##### ➕ Create / Update User")
                
                # Inputs
                u_new = st.text_input("Username (Unique)")
                p_new = st.text_input("Set Password", type="password")
                n_new = st.text_input("Display Name")
                r_new = st.selectbox("Assign Role Label", ["admin", "manager", "sales"])
                
                # ACCESS RIGHTS SELECTION
                st.markdown("**Assign Access (Select Tabs):**")
                default_access = ALL_MODULES if r_new == 'admin' else ["Dashboard"]
                a_new = st.multiselect("Allowed Tabs", ALL_MODULES, default=default_access)
                
                if st.button("Create/Update User"):
                    if u_new and p_new and n_new and a_new:
                        users = load_users()
                        users[u_new] = {
                            "password": p_new, 
                            "role": r_new, 
                            "name": n_new,
                            "access": a_new # Saving specific permissions
                        }
                        save_users(users)
                        st.success(f"✅ User '{u_new}' Saved with {len(a_new)} tabs!")
                    else:
                        st.error("All fields & at least one tab required.")
                
                st.markdown("---")
                st.markdown("##### 🗑️ Delete User")
                users = load_users()
                del_user = st.selectbox("Select User to Delete", list(users.keys()))
                if st.button("Delete User", type="primary"):
                    if del_user == st.session_state['user']:
                        st.error("❌ You cannot delete yourself!")
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
    
    # Constants
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

    # Load Data
    def get_file_timestamp():
        if os.path.exists(FILE_PATH): return os.path.getmtime(FILE_PATH)
        return 0

    @st.cache_data
    def load_data(last_modified_time):
        if not os.path.exists(FILE_PATH): return None
        try:
            df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
            df.columns = df.columns.str.strip()
            if 'Invoice Date' in df.columns:
                df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], dayfirst=True, errors='coerce')
            if 'Chassis No.' in df.columns:
                df['Chassis No.'] = df['Chassis No.'].astype(str)
            
            target_cols = ['Sale Invoice Amount With GST', 'Sale Invoice Amount Basic Value', 'Purchase With GST Value', 'Purchase Basic Value', 'TOTAL OEM DISCOUNTS', 'TOTAL INTENAL DISCOUNTS', 'TOTAL OEM & INTERNAL NET DISCOUNTS', 'TOTAL Credit Note NET DISCOUNT', 'MARGIN', 'TOTAL RECEIVED OEM NET DISCOUNTS', 'FINAL MARGIN', 'OEM - RETAIL SCHEME', 'RECEIVED OEM - RETAIL SCHEME', 'OEM - CORPORATE SCHEME', 'RECEIVED OEM - CORPORATE SCHEME', 'OEM - EXCHANGE SCHEME', 'RECEIVED OEM - EXCHANGE SCHEME', 'OEM - SPECIAL SCHEME', 'RECEIVED OEM - SPECIAL SCHEME', 'OEM - WHOLESALE SUPPORT', 'RECEIVED OEM - WHOLESALE SUPPORT', 'OEM - LOYALTY BONUS', 'RECEIVED OEM - LOYALTY BONUS', 'OEM - OTHERS', 'RECEIVED OEM - OTHERS', 'TOTAL Credit Note Amout OEM']
            
            for col in target_cols:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                else: df[col] = 0
            return df
        except Exception as e:
            st.error(f"Error: {e}"); return None

    def to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer: df.to_excel(writer, index=False, sheet_name='Report')
        return output.getvalue()

    # Helpers
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
    
    def clear_specific_file(file_path):
        if os.path.exists(file_path): os.remove(file_path); return True
        return False

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
        
        # --- ADD AVG MARGIN COLUMN ---
        if "FINAL MARGIN" in final.columns and "Count" in final.columns:
            final["Avg Margin"] = (final["FINAL MARGIN"] / final["Count"]).fillna(0)
        
        total_row = {c: "" for c in valid_cols}; total_row[valid_cols[0]] = "GRAND TOTAL"; total_row["Count"] = final['Count'].sum()
        for col in metrics.values(): total_row[col] = final[col].sum()
        
        total_row["Avg Margin"] = total_row["FINAL MARGIN"] / total_row["Count"] if total_row["Count"] > 0 else 0
        
        avg_row = {c: "" for c in valid_cols}; avg_row[valid_cols[0]] = "AVERAGE PER VEHICLE"; avg_row["Count"] = 1
        for col in metrics.values(): avg_row[col] = total_row[col] / total_row["Count"] if total_row["Count"] > 0 else 0
        avg_row["Avg Margin"] = total_row["Avg Margin"]
        
        final = pd.concat([final, pd.DataFrame([total_row]), pd.DataFrame([avg_row])], ignore_index=True)
        final.rename(columns={v: k for k, v in metrics.items()}, inplace=True)
        
        # Format mapping for Styler
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
        
        # PIVOT - COUNTING RECORDS
        pivot_data = df.pivot_table(index=group_cols, columns='Month_Sort', values=date_col, aggfunc='count', fill_value=0)
        
        if start_date and end_date:
            expected_months = pd.period_range(start=start_date, end=end_date, freq='M')
            pivot_data = pivot_data.reindex(columns=expected_months, fill_value=0)

        pivot_data['Total'] = pivot_data.sum(axis=1)
        month_count = len(pivot_data.columns) - 1
        pivot_data['Average'] = pivot_data['Total'] / month_count if month_count > 0 else 0
        
        gt_row = pivot_data.sum(axis=0)
        if month_count > 0: gt_row['Average'] = gt_row['Total'] / month_count
        
        # --- FIX FOR SINGLE COLUMN VS MULTI COLUMN GROUPING ---
        if len(group_cols) > 1:
            gt_name = tuple(['GRAND TOTAL'] + [''] * (len(group_cols) - 1))
        else:
            gt_name = 'GRAND TOTAL'
            
        pivot_data.loc[gt_name, :] = gt_row
        
        pivot_data.columns = [c.strftime('%b-%Y') if isinstance(c, pd.Period) else c for c in pivot_data.columns]
        return pivot_data

    # --- HELPER FOR IMAGE ENCODING ---
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # --- APP START ---
    
    # --- HEADER WITH LOGOS ---
    logo_left_path = "logo_left.png"
    logo_right_path = "logo_right.png"
    logo_left_html = ""
    logo_right_html = ""

    # CSS style for logos (Same Size, No Stretching)
    img_style = "height: 80px; width: auto; object-fit: contain; max-width: 150px;"

    if os.path.exists(logo_left_path):
        logo_left_b64 = get_base64_of_bin_file(logo_left_path)
        logo_left_html = f'<img src="data:image/png;base64,{logo_left_b64}" style="{img_style}">'

    if os.path.exists(logo_right_path):
        logo_right_b64 = get_base64_of_bin_file(logo_right_path)
        logo_right_html = f'<img src="data:image/png;base64,{logo_right_b64}" style="{img_style}">'

    # Flexbox container for header
    header_html = f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0;">
        <div style="flex: 1; text-align: left;">
            {logo_left_html}
        </div>
        <div style="flex: 6; text-align: center;">
            <h1 style="margin: 0;">🚗 Vehicle Sales Management System</h1>
        </div>
        <div style="flex: 1; text-align: right;">
            {logo_right_html}
        </div>
    </div>
    <hr style="margin-top: 0; margin-bottom: 20px;">
    """
    st.markdown(header_html, unsafe_allow_html=True)
    
    # --- END HEADER ---

    ts = get_file_timestamp()
    
    if ts == 0:
        st.error("❌ File not found!")
    else:
        df = load_data(ts)
        if df is not None:
            
            # --- GET ALLOWED TABS FROM USER DB ---
            allowed_tabs = st.session_state.get('access', [])
            
            # Fallback if no tabs assigned
            if not allowed_tabs:
                allowed_tabs = ["Dashboard"]

            # 3. Create Dynamic Tabs
            tabs = st.tabs(allowed_tabs)
            
            # 4. Map Tab Name to Tab Object
            tab_map = {name: tab for name, tab in zip(allowed_tabs, tabs)}

            # --- RENDER TAB CONTENT ---
            
            # TAB: DASHBOARD
            if "Dashboard" in tab_map:
                with tab_map["Dashboard"]:
                    st.subheader("Overview")
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Total Vehicles", format_lakhs(len(df)))
                    k2.metric("Total Revenue", f"₹ {format_lakhs(df['Sale Invoice Amount With GST'].sum())}")
                    k3.metric("Total Final Margin", f"₹ {format_lakhs(df['FINAL MARGIN'].sum())}")
                    st.dataframe(df)

            # TAB: SEARCH & EDIT (DROPDOWN STYLE - UPDATED)
            if "Search & Edit" in tab_map:
                with tab_map["Search & Edit"]:
                    st.header("Search & Edit Records")
                    
                    # 1. Search Bar (Dropdown for Mode)
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
                            idx = res.index[0] # Editing the first match
                            
                            st.markdown("### 📂 Update Records")
                            
                            # Helper to safely save
                            def save_changes(new_data_dict):
                                try:
                                    for c, v in new_data_dict.items():
                                        if c not in df.columns: df[c] = None # Create col if missing
                                        df.at[idx, c] = v
                                    df.to_excel(FILE_PATH, sheet_name=SHEET_NAME, index=False)
                                    st.success("✅ Saved Successfully!")
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")

                            # --- CATEGORY DROPDOWN ---
                            update_cat = st.selectbox(
                                "Select Category to Update:", 
                                ["Select an option", "Sale Updation", "Discount Updation", "HSRP Updation", "Finance Updation", "Insurance Updation"]
                            )

                            # --- 1. SALE UPDATION ---
                            if update_cat == "Sale Updation":
                                st.subheader("📝 Sale Details")
                                with st.form("sale_form"):
                                    sale_cols = [
                                        "Model", "Variant", "Colour", "Chassis No.", "Engine No", 
                                        "Customer Name", "Employee Code (HRMS)", "Sales Consultant Name", 
                                        "Month Wise FSC Target", "ASM", "SM", "Outlet"
                                    ]
                                    s_data = {}
                                    cols = st.columns(3)
                                    for i, col in enumerate(sale_cols):
                                        val = str(df.at[idx, col]) if col in df.columns and pd.notna(df.at[idx, col]) else ""
                                        with cols[i % 3]:
                                            s_data[col] = st.text_input(col, value=val)
                                    
                                    if st.form_submit_button("💾 Save Sale Details"):
                                        save_changes(s_data)

                            # --- 2. DISCOUNT UPDATION ---
                            elif update_cat == "Discount Updation":
                                st.subheader("💰 Discount Details")
                                t_oem, t_int, t_cn = st.tabs(["🔹 OEM Discounts", "🔹 Internal Discounts", "🔹 Credit Note"])
                                
                                # Window 1: OEM
                                with t_oem:
                                    with st.form("oem_form"):
                                        oem_cols = [
                                            "OEM - RETAIL SCHEME", "OEM - CORPORATE SCHEME", "OEM - EXCHANGE SCHEME",
                                            "OEM - SPECIAL SCHEME", "OEM - WHOLESALE SUPPORT", "OEM - LOYALTY BONUS",
                                            "OEM - OTHERS", "TOTAL OEM DISCOUNTS"
                                        ]
                                        o_data = {}
                                        c_oem = st.columns(3)
                                        for i, col in enumerate(oem_cols):
                                            val = str(df.at[idx, col]) if col in df.columns and pd.notna(df.at[idx, col]) else ""
                                            with c_oem[i % 3]:
                                                o_data[col] = st.text_input(col, value=val)
                                        if st.form_submit_button("💾 Save OEM Discounts"):
                                            save_changes(o_data)

                                # Window 2: Internal
                                with t_int:
                                    with st.form("int_form"):
                                        int_cols = [
                                            "INTERNAL - RETAIL SCHEME", "INTERNAL - CORPORATE SCHEME", "INTERNAL - EXCHANGE SUPPORT",
                                            "INTERNAL - Accesories Discount", "INTERNAL - Dealer Cash Discount", "INTERNAL - Employee Discount",
                                            "INTERNAL - Referal Bonus", "INTERNAL - EW Scheme", "INTERNAL - Depreciation",
                                            "INTERNAL - Other discounts", "INTERNAL - Additional Special discount", "INTERNAL - Loyalty Scheme",
                                            "TOTAL INTENAL DISCOUNTS"
                                        ]
                                        i_data = {}
                                        c_int = st.columns(3)
                                        for i, col in enumerate(int_cols):
                                            val = str(df.at[idx, col]) if col in df.columns and pd.notna(df.at[idx, col]) else ""
                                            with c_int[i % 3]:
                                                i_data[col] = st.text_input(col, value=val)
                                        if st.form_submit_button("💾 Save Internal Discounts"):
                                            save_changes(i_data)

                                # Window 3: Credit Note
                                with t_cn:
                                    with st.form("cn_form"):
                                        cn_cols = [
                                            "TOTAL Credit Note Amout OEM", "TOTAL Credit Note Amout INTERNAL", "TOTAL Credit Note NET DISCOUNT", "MARGIN",
                                            "RECEIVED OEM - RETAIL SCHEME", "RECEIVED OEM - CORPORATE SCHEME", "RECEIVED OEM - EXCHANGE SCHEME",
                                            "RECEIVED OEM - SPECIAL SCHEME", "RECEIVED OEM - WHOLESALE SUPPORT", "RECEIVED OEM - LOYALTY BONUS",
                                            "RECEIVED OEM - OTHERS", "TOTAL RECEIVED OEM NET DISCOUNTS", "RECEIVED OEM REMARKS (IF ANY REASON OR CREDIT NOTE NO)",
                                            "FINAL MARGIN", "Credit Note Reference No", "Credit Note Reference Date"
                                        ]
                                        cn_data = {}
                                        c_cn = st.columns(3)
                                        for i, col in enumerate(cn_cols):
                                            val = str(df.at[idx, col]) if col in df.columns and pd.notna(df.at[idx, col]) else ""
                                            with c_cn[i % 3]:
                                                cn_data[col] = st.text_input(col, value=val)
                                        if st.form_submit_button("💾 Save Credit Note Info"):
                                            save_changes(cn_data)

                            # --- 3. HSRP UPDATION ---
                            elif update_cat == "HSRP Updation":
                                st.subheader("🚦 HSRP Details")
                                t_tr, t_pr = st.tabs(["🔸 TR Updation", "🔸 PR Updation"])
                                
                                with t_tr:
                                    with st.form("tr_form"):
                                        tr_cols = ["TR Date", "TR Number", "Application Numebr", "RTA NAME", "TR Amount"]
                                        tr_data = {}
                                        c_tr = st.columns(3)
                                        for i, col in enumerate(tr_cols):
                                            val = str(df.at[idx, col]) if col in df.columns and pd.notna(df.at[idx, col]) else ""
                                            with c_tr[i % 3]:
                                                tr_data[col] = st.text_input(col, value=val)
                                        if st.form_submit_button("💾 Save TR Details"):
                                            save_changes(tr_data)
                                
                                with t_pr:
                                    with st.form("pr_form"):
                                        pr_cols = [
                                            "PR Number", "PR REGISTRATION DATE", "MGF DATE", "FUEL", "VEHICLE CLASS",
                                            "PR Ordered Date", "PR Ordered Status", "PR Ordered Amount", "PR STATUS REMARKS"
                                        ]
                                        pr_data = {}
                                        c_pr = st.columns(3)
                                        for i, col in enumerate(pr_cols):
                                            val = str(df.at[idx, col]) if col in df.columns and pd.notna(df.at[idx, col]) else ""
                                            with c_pr[i % 3]:
                                                pr_data[col] = st.text_input(col, value=val)
                                        if st.form_submit_button("💾 Save PR Details"):
                                            save_changes(pr_data)

                            # --- 4. FINANCE UPDATION ---
                            elif update_cat == "Finance Updation":
                                st.subheader("💳 Finance Details")
                                with st.form("fin_form"):
                                    fin_cols = [
                                        "FINANCE IN/OUT", "Name of the Financier", "Laon Amount", 
                                        "% of payout", "Finance Payout receivable"
                                    ]
                                    f_data = {}
                                    c_fin = st.columns(3)
                                    for i, col in enumerate(fin_cols):
                                        val = str(df.at[idx, col]) if col in df.columns and pd.notna(df.at[idx, col]) else ""
                                        with c_fin[i % 3]:
                                            f_data[col] = st.text_input(col, value=val)
                                    if st.form_submit_button("💾 Save Finance Details"):
                                        save_changes(f_data)

                            # --- 5. INSURANCE UPDATION ---
                            elif update_cat == "Insurance Updation":
                                st.subheader("🛡️ Insurance Details")
                                with st.form("ins_form"):
                                    ins_cols = ["INSURANCE IN/OUT", "INS DISCOUNT %", "Policy NO", "Insurance Company Name"]
                                    ins_data = {}
                                    c_ins = st.columns(2)
                                    for i, col in enumerate(ins_cols):
                                        val = str(df.at[idx, col]) if col in df.columns and pd.notna(df.at[idx, col]) else ""
                                        with c_ins[i % 2]:
                                            ins_data[col] = st.text_input(col, value=val)
                                    if st.form_submit_button("💾 Save Insurance Details"):
                                        save_changes(ins_data)

                        else: st.warning("No records found.")

            # TAB: FINANCIAL REPORTS
            if "Financial Reports" in tab_map:
                with tab_map["Financial Reports"]:
                    st.header("📈 Financial Reports")
                    min_d = df['Invoice Date'].min().date() if 'Invoice Date' in df.columns else None
                    max_d = df['Invoice Date'].max().date() if 'Invoice Date' in df.columns else None
                    c1, c2 = st.columns(2)
                    start_date = c1.date_input("From Date", value=min_d, key="fin_start")
                    end_date = c2.date_input("To Date", value=max_d, key="fin_end")
                    mask = (df['Invoice Date'].dt.date >= start_date) & (df['Invoice Date'].dt.date <= end_date)
                    report_df = df.loc[mask]
                    group_by = st.selectbox("View:", ["Segment", "Sales Consultant Name", "Outlet", "ASM", "Model"], key="fin_grp")
                    
                    if group_by in report_df.columns:
                        generate_all_report(report_df, group_by, f"{group_by} Financial Performance")

            # TAB: OEM PENDING
            if "OEM Pending Analysis" in tab_map:
                with tab_map["OEM Pending Analysis"]:
                    st.header("📉 OEM Pending vs Received Report")
                    
                    # Date Selection
                    min_d = df['Invoice Date'].min().date() if 'Invoice Date' in df.columns else None
                    max_d = df['Invoice Date'].max().date() if 'Invoice Date' in df.columns else None
                    c3, c4 = st.columns(2)
                    p_start_date = c3.date_input("From Date", value=min_d, key="p_start")
                    p_end_date = c4.date_input("To Date", value=max_d, key="p_end")
                    
                    mask = (df['Invoice Date'].dt.date >= p_start_date) & (df['Invoice Date'].dt.date <= p_end_date)
                    p_df = df.loc[mask].copy()
                    
                    # Scheme Definitions
                    scheme_pairs = [
                        ('OEM - RETAIL SCHEME', 'RECEIVED OEM - RETAIL SCHEME', 'Pending Retail'),
                        ('OEM - CORPORATE SCHEME', 'RECEIVED OEM - CORPORATE SCHEME', 'Pending Corporate'),
                        ('OEM - EXCHANGE SCHEME', 'RECEIVED OEM - EXCHANGE SCHEME', 'Pending Exchange'),
                        ('OEM - SPECIAL SCHEME', 'RECEIVED OEM - SPECIAL SCHEME', 'Pending Special'),
                        ('OEM - WHOLESALE SUPPORT', 'RECEIVED OEM - WHOLESALE SUPPORT', 'Pending Wholesale'),
                        ('OEM - LOYALTY BONUS', 'RECEIVED OEM - LOYALTY BONUS', 'Pending Loyalty'),
                        ('OEM - OTHERS', 'RECEIVED OEM - OTHERS', 'Pending Others')
                    ]
                    
                    # Core Calculations
                    p_df['PENDING_TOTAL'] = p_df['TOTAL OEM DISCOUNTS'] - p_df['TOTAL RECEIVED OEM NET DISCOUNTS']
                    p_df['STATUS'] = p_df['PENDING_TOTAL'].apply(lambda x: "PENDING" if x > 1 else "RECEIVED/CLEARED")
                    
                    base_cols = ['Chassis No.', 'Customer Name', 'Invoice No.', 'Invoice Date', 'Model', 'Outlet', 'Sales Consultant Name']
                    valid_base = [c for c in base_cols if c in p_df.columns]
                    
                    # Pending Export Prep
                    pending_export_df = p_df[p_df['STATUS'] == "PENDING"].copy()
                    pending_calc_cols = []
                    for given, received, pending_name in scheme_pairs:
                        if given in pending_export_df.columns and received in pending_export_df.columns:
                            pending_export_df[pending_name] = pending_export_df[given] - pending_export_df[received]
                            pending_calc_cols.append(pending_name)
                    
                    if not pending_export_df.empty:
                        final_p_cols = valid_base + pending_calc_cols + ['PENDING_TOTAL']
                        pending_final_export = pending_export_df[final_p_cols].copy()
                        pending_final_export = pending_final_export.loc[:, (pending_final_export != 0).any(axis=0)]
                    else:
                        pending_final_export = pd.DataFrame()

                    # Received Export Prep
                    received_export_df = p_df[p_df['STATUS'] == "RECEIVED/CLEARED"].copy()
                    received_cols_only = [pair[1] for pair in scheme_pairs if pair[1] in received_export_df.columns]
                    
                    if not received_export_df.empty:
                        final_r_cols = valid_base + received_cols_only + ['TOTAL RECEIVED OEM NET DISCOUNTS']
                        received_final_export = received_export_df[final_r_cols].copy()
                        received_final_export = received_final_export.loc[:, (received_final_export != 0).any(axis=0)]
                    else:
                        received_final_export = pd.DataFrame()

                    # --- NEW ENHANCEMENTS (KPIs + Charts) ---
                    
                    # 1. KPI Cards
                    tot_pend = p_df[p_df['STATUS']=="PENDING"]['PENDING_TOTAL'].sum()
                    tot_rec = p_df['TOTAL RECEIVED OEM NET DISCOUNTS'].sum()
                    rec_rate = (tot_rec / p_df['TOTAL OEM DISCOUNTS'].sum() * 100) if p_df['TOTAL OEM DISCOUNTS'].sum() > 0 else 0
                    
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Total Pending Amount", f"₹ {format_lakhs(tot_pend)}")
                    k2.metric("Total Received Amount", f"₹ {format_lakhs(tot_rec)}")
                    k3.metric("Recovery Rate", f"{rec_rate:.1f}%")
                    
                    st.markdown("---")

                    # Display Counts & Download Buttons
                    d1, d2 = st.columns(2)
                    with d1:
                        st.write(f"**🔴 Pending: {len(pending_final_export)}**")
                        if not pending_final_export.empty:
                            st.download_button("Download PENDING List", data=to_excel(pending_final_export), file_name="Detailed_Pending_List.xlsx")
                    with d2:
                        st.write(f"**🟢 Received: {len(received_final_export)}**")
                        if not received_final_export.empty:
                            st.download_button("Download RECEIVED List", data=to_excel(received_final_export), file_name="Detailed_Received_List.xlsx")
                    
                    # 2. Scheme Wise Summary & Charts
                    st.markdown("---")
                    st.subheader("📊 Analysis & Charts")
                    
                    c_chart1, c_chart2 = st.columns(2)
                    
                    # Prepare Scheme Data
                    scheme_data = []
                    for given, received, pending_name in scheme_pairs:
                        if given in p_df.columns and received in p_df.columns:
                            pend_amt = (p_df[given] - p_df[received]).sum()
                            scheme_data.append({"Scheme Type": pending_name.replace("Pending ", ""), "Pending Amount": pend_amt})
                    
                    scheme_df = pd.DataFrame(scheme_data)
                    
                    with c_chart1:
                        st.markdown("**Pending by Scheme Type**")
                        if not scheme_df.empty:
                            fig = px.pie(scheme_df, values='Pending Amount', names='Scheme Type', hole=0.4)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Custom format for Scheme Table
                            st.dataframe(scheme_df.style.format({"Pending Amount": lambda x: f"₹ {format_lakhs(x)}"}))

                    # Group Analysis & Bar Chart
                    with c_chart2:
                        st.markdown("**Pending vs Received (Group Wise)**")
                        p_group_by = st.selectbox("Group By:", ["Segment", "Model", "Outlet", "Sales Consultant Name"], key="p_group")
                        
                        if p_group_by in p_df.columns:
                            p_grouped = p_df.copy()
                            p_grouped[p_group_by] = p_grouped[p_group_by].fillna("Unknown")
                            given_col = "TOTAL OEM DISCOUNTS"
                            received_col = "TOTAL RECEIVED OEM NET DISCOUNTS"
                            
                            p_report = p_grouped.groupby(p_group_by)[[received_col, "PENDING_TOTAL"]].sum().reset_index()
                            p_report.rename(columns={received_col: "Received", "PENDING_TOTAL": "Pending"}, inplace=True)
                            
                            # Bar Chart
                            fig_bar = px.bar(p_report, x=p_group_by, y=["Received", "Pending"], barmode='group')
                            st.plotly_chart(fig_bar, use_container_width=True)

                            # Data Table
                            p_report["Total Given"] = p_report["Received"] + p_report["Pending"]
                            p_report["Recovery %"] = (p_report["Received"] / p_report["Total Given"] * 100).fillna(0)
                            
                            # Apply Indian Format to Money Columns in Pending Report
                            st.dataframe(p_report.style.format({
                                "Received": lambda x: f"₹ {format_lakhs(x)}", 
                                "Pending": lambda x: f"₹ {format_lakhs(x)}", 
                                "Total Given": lambda x: f"₹ {format_lakhs(x)}", 
                                "Recovery %": "{:.1f}%"
                            }))

                    # --- 3. CREDIT NOTE ANALYSIS ---
                    st.markdown("---")
                    st.subheader("💳 OEM Credit Note Analysis")
                    
                    cn_col = "TOTAL Credit Note Amout OEM"
                    
                    if cn_col in p_df.columns:
                        cn_df = p_df[p_df[cn_col] > 0].copy()
                        
                        if not cn_df.empty:
                            cn_total_amt = cn_df[cn_col].sum()
                            cn_count = len(cn_df)
                            
                            m1, m2 = st.columns(2)
                            m1.metric("Total Credit Note Amount", f"₹ {format_lakhs(cn_total_amt)}")
                            m2.metric("Total Credit Note Count", format_lakhs(cn_count))
                            
                            disp_cols = valid_base + [cn_col, 'Credit Note Reference No', 'Credit Note Reference Date', 'RECEIVED OEM REMARKS (IF ANY REASON OR CREDIT NOTE NO)']
                            disp_cols = [c for c in disp_cols if c in cn_df.columns]
                            
                            cn_disp_df = cn_df[disp_cols]
                            # Format dataframe for display
                            st.dataframe(cn_disp_df.style.format({cn_col: lambda x: f"₹ {format_lakhs(x)}"}))
                            
                            st.download_button(
                                "Download Credit Note Details", 
                                data=to_excel(cn_disp_df), 
                                file_name="OEM_Credit_Note_Details.xlsx"
                            )
                        else:
                            st.info("No records found with Credit Note Amount > 0 in the selected period.")
                    else:
                        st.error(f"Column '{cn_col}' not found in data.")

                    # --- 4. SCHEME WISE PERFORMANCE ANALYSIS (ENHANCED - COLOR CODED EXCEL & SUMMARY) ---
                    st.markdown("---")
                    st.subheader("📑 Scheme Wise Performance (Given vs Received vs Pending)")

                    # Calculate Summary
                    summary_data = []
                    for given, received, pending_name in scheme_pairs:
                        if given in p_df.columns and received in p_df.columns:
                            s_given = p_df[given].sum()
                            s_rec = p_df[received].sum()
                            s_pend = s_given - s_rec
                            
                            # Determine Status
                            if s_pend > 0:
                                status = "Shortage"
                            elif s_pend < 0:
                                status = "Excess"
                            else:
                                status = "Balanced"
                                
                            s_rec_pct = (s_rec / s_given * 100) if s_given > 0 else 0
                            
                            summary_data.append({
                                "Scheme Type": pending_name.replace("Pending ", ""),
                                "Total OEM Discounts": s_given,
                                "Actual OEM Received": s_rec,
                                "Pending/Excess Amount": s_pend,
                                "Status": status,
                                "Recovery %": s_rec_pct
                            })

                    if summary_data:
                        summ_df = pd.DataFrame(summary_data)
                        
                        # Grand Total Row
                        gt_g = summ_df["Total OEM Discounts"].sum()
                        gt_r = summ_df["Actual OEM Received"].sum()
                        gt_p = summ_df["Pending/Excess Amount"].sum()
                        gt_pct = (gt_r / gt_g * 100) if gt_g > 0 else 0
                        
                        gt_row = pd.DataFrame([{
                            "Scheme Type": "GRAND TOTAL",
                            "Total OEM Discounts": gt_g,
                            "Actual OEM Received": gt_r,
                            "Pending/Excess Amount": gt_p,
                            "Status": "-",
                            "Recovery %": gt_pct
                        }])
                        
                        summ_df = pd.concat([summ_df, gt_row], ignore_index=True)
                        
                        # Display Summary Table with Styling
                        def highlight_status(val):
                            if val == 'Shortage': return 'color: red; font-weight: bold'
                            elif val == 'Excess': return 'color: green; font-weight: bold'
                            return ''

                        st.dataframe(summ_df.style.format({
                            "Total OEM Discounts": lambda x: f"₹ {format_lakhs(x)}",
                            "Actual OEM Received": lambda x: f"₹ {format_lakhs(x)}",
                            "Pending/Excess Amount": lambda x: f"₹ {format_lakhs(x)}",
                            "Recovery %": "{:.1f}%"
                        }).applymap(highlight_status, subset=['Status'])
                          .apply(lambda x: ['background-color: #f0f0f0; font-weight: bold' if x['Scheme Type'] == 'GRAND TOTAL' else '' for _ in x], axis=1))

                        # Download Detailed Report (WITH COLOR CODING)
                        # Prepare detailed DataFrame
                        det_cols = valid_base.copy()
                        detailed_df = p_df[det_cols].copy()
                        
                        pending_cols_list = []
                        received_cols_list = []
                        given_cols_list = []
                        
                        for given, received, pending_name in scheme_pairs:
                            if given in p_df.columns and received in p_df.columns:
                                s_name = pending_name.replace("Pending ", "")
                                g_col = f"{s_name} - Given"
                                r_col = f"{s_name} - Received"
                                p_col = f"{s_name} - Pending"
                                
                                detailed_df[g_col] = p_df[given]
                                detailed_df[r_col] = p_df[received]
                                detailed_df[p_col] = p_df[given] - p_df[received]
                                
                                given_cols_list.append(g_col)
                                received_cols_list.append(r_col)
                                pending_cols_list.append(p_col)
                                
                        # Add Totals
                        detailed_df["TOTAL GIVEN"] = p_df["TOTAL OEM DISCOUNTS"]
                        detailed_df["TOTAL RECEIVED"] = p_df["TOTAL RECEIVED OEM NET DISCOUNTS"]
                        detailed_df["TOTAL DIFFERENCE AMOUNT"] = p_df["PENDING_TOTAL"]
                        
                        pending_cols_list.append("TOTAL DIFFERENCE AMOUNT")
                        received_cols_list.append("TOTAL RECEIVED")
                        given_cols_list.append("TOTAL GIVEN")

                        # Function to create styled Excel
                        def to_styled_excel(df):
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                # Apply styling
                                styler = df.style
                                # Highlight Pending/Difference columns (Red text if > 0)
                                styler.applymap(lambda v: 'color: red; font-weight: bold' if isinstance(v, (int, float)) and v > 0 else '', subset=pending_cols_list)
                                # Highlight Received columns (Green text)
                                styler.applymap(lambda v: 'color: green' if isinstance(v, (int, float)) and v > 0 else '', subset=received_cols_list)
                                # Highlight Given columns (Blue text)
                                styler.applymap(lambda v: 'color: blue', subset=given_cols_list)
                                
                                styler.to_excel(writer, index=False, sheet_name='Scheme_Wise_Report')
                            return output.getvalue()
                        
                        st.download_button(
                            "Download Detailed Scheme Wise Report (Color Coded)",
                            data=to_styled_excel(detailed_df),
                            file_name="Detailed_Scheme_Wise_Report.xlsx"
                        )

            # TAB: DATA QUALITY CHECK (NEW - ADDED BACK)
            if "Data Quality Check" in tab_map:
                with tab_map["Data Quality Check"]:
                    st.header("🛡️ Data Quality Inspector")
                    st.info("This tool scans your data for common errors like missing values, duplicates, and negative amounts.")
                    
                    if st.button("Run Quality Check", type="primary"):
                        errors = []
                        
                        # 1. Duplicate Chassis
                        if 'Chassis No.' in df.columns:
                            dupes = df[df.duplicated('Chassis No.', keep=False)]
                            if not dupes.empty:
                                for i, row in dupes.iterrows():
                                    errors.append({"Row No": i+2, "Issue Type": "Duplicate Chassis", "Description": f"Chassis {row['Chassis No.']} is repeated.", "Value": row['Chassis No.']})
                        
                        # 2. Missing Mandatory Fields
                        mandatory_cols = ['Invoice Date', 'Customer Name', 'Model', 'Outlet', 'Sales Consultant Name']
                        for col in mandatory_cols:
                            if col in df.columns:
                                missing = df[df[col].isna() | (df[col] == '')]
                                for i, row in missing.iterrows():
                                    errors.append({"Row No": i+2, "Issue Type": "Missing Data", "Description": f"Column '{col}' is empty.", "Value": "Empty"})
                        
                        # 3. Negative Values (Money Columns)
                        money_cols = [c for c in df.columns if any(x in c.upper() for x in ['AMOUNT', 'PRICE', 'VALUE', 'MARGIN', 'DISCOUNT'])]
                        for col in money_cols:
                            # Ensure numeric
                            if pd.api.types.is_numeric_dtype(df[col]):
                                negatives = df[df[col] < 0]
                                # Filter out "Margin" or "Discount" if negatives are allowed there (Usually discounts are positive in data, margin can be neg)
                                # Let's flag all negatives for review
                                for i, row in negatives.iterrows():
                                    errors.append({"Row No": i+2, "Issue Type": "Negative Value", "Description": f"Column '{col}' has negative value.", "Value": row[col]})

                        if errors:
                            err_df = pd.DataFrame(errors)
                            st.error(f"Found {len(errors)} issues!")
                            st.dataframe(err_df)
                            st.download_button("Download Error Report", data=to_excel(err_df), file_name="Data_Quality_Errors.xlsx")
                        else:
                            st.success("✅ No Data Quality Issues Found! Great Job!")

            # TAB: TALLY & TOS
            if "Tally & TOS Reports" in tab_map:
                with tab_map["Tally & TOS Reports"]:
                    st.header("📑 Tally & TOS Registers")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.file_uploader("Upload Tally Sale", key="u1"): save_and_append(st.session_state.u1, FILES_DB["Tally Sale"])
                        if st.file_uploader("Upload Tally Purchase", key="u2"): save_and_append(st.session_state.u2, FILES_DB["Tally Purchase"])
                    with c2:
                        if st.file_uploader("Upload TOS In", key="u3"): save_and_append(st.session_state.u3, FILES_DB["TOS In"])
                        if st.file_uploader("Upload TOS Out", key="u4"): save_and_append(st.session_state.u4, FILES_DB["TOS Out"])
                    
                    if st.button("Clean Duplicates"): 
                        remove_duplicates_only(FILES_DB["Tally Sale"])
                        st.success("Cleaned!")

            # TAB: ALL REPORT (DROPDOWN NAVIGATION)
            if "All Report" in tab_map:
                with tab_map["All Report"]:
                    st.header("📋 Consolidated Reports")
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
                        
                        # Helper for consolidation logic (moved up for scope availability)
                        def create_consolidated_report(grp_cols):
                            if not grp_cols: return pd.DataFrame()
                            
                            df_cons = all_rep_df.copy()
                            for col in grp_cols:
                                df_cons[col] = df_cons[col].fillna("Unknown")
                            
                            fin_col = "FINANCE IN/OUT"
                            ins_col = "INSURANCE IN/OUT"
                            
                            if fin_col in df_cons.columns:
                                df_cons['Fin_Type'] = df_cons[fin_col].astype(str).str.strip().str.upper()
                                df_cons['Fin_Name'] = df_cons['Name of the Financier'].astype(str).str.strip().str.upper() if 'Name of the Financier' in df_cons.columns else ''
                                is_in_house = df_cons['Fin_Type'] == 'IN HOUSE'
                                is_out_house = df_cons['Fin_Type'] == 'OUT HOUSE'
                                is_own_lease = df_cons['Fin_Type'].isin(['OWN FUNDS', 'LEASE'])
                                is_mmfsl = is_in_house & df_cons['Fin_Name'].str.contains('MAHINDRA', case=False)
                                is_other_in = is_in_house & ~is_mmfsl
                                df_cons['Other_Fin_In'] = is_other_in.astype(int)
                                df_cons['MMFSL_Fin_In'] = is_mmfsl.astype(int)
                                df_cons['Total_In_House'] = is_in_house.astype(int)
                                df_cons['Own_Lease'] = is_own_lease.astype(int)
                                df_cons['Total_Out_House'] = is_out_house.astype(int)
                                df_cons['Total_Fin_InOut'] = (is_in_house | is_out_house).astype(int)
                            
                            if ins_col in df_cons.columns:
                                df_cons['Ins_Type'] = df_cons[ins_col].astype(str).str.strip().str.upper()
                                df_cons['Ins_In'] = (df_cons['Ins_Type'] == 'IN HOUSE').astype(int)
                                df_cons['Ins_Out'] = (df_cons['Ins_Type'] == 'OUT HOUSE').astype(int)
                                
                            df_cons['Tally_Sale_Count'] = 1
                            
                            agg_dict = {
                                'Other_Fin_In': 'sum', 'MMFSL_Fin_In': 'sum', 'Total_In_House': 'sum',
                                'Own_Lease': 'sum', 'Total_Out_House': 'sum', 'Total_Fin_InOut': 'sum',
                                'Tally_Sale_Count': 'sum', 'Ins_In': 'sum', 'Ins_Out': 'sum'
                            }
                            agg_dict = {k: v for k, v in agg_dict.items() if k in df_cons.columns}
                            
                            grouped_res = df_cons.groupby(grp_cols).agg(agg_dict)
                            
                            if 'Total_In_House' in grouped_res.columns:
                                grouped_res['Finance Grand Total'] = (grouped_res['Total_In_House'] + grouped_res.get('Own_Lease', 0) + grouped_res.get('Total_Out_House', 0))
                            
                            if 'Tally_Sale_Count' in grouped_res.columns and 'Finance Grand Total' in grouped_res.columns:
                                grouped_res['Difference'] = grouped_res['Tally_Sale_Count'] - grouped_res['Finance Grand Total']

                            if 'Ins_In' in grouped_res.columns:
                                grouped_res['Insurance Grand Total'] = grouped_res['Ins_In'] + grouped_res.get('Ins_Out', 0)

                            if 'Total_In_House' in grouped_res.columns and 'Total_Fin_InOut' in grouped_res.columns:
                                grouped_res['Fin_In_House_Pct'] = (grouped_res['Total_In_House'] / grouped_res['Total_Fin_InOut'] * 100).fillna(0)
                            
                            if 'MMFSL_Fin_In' in grouped_res.columns and 'Total_In_House' in grouped_res.columns:
                                grouped_res['MMFSL_Share_Pct'] = (grouped_res['MMFSL_Fin_In'] / grouped_res['Total_In_House'] * 100).fillna(0)
                                
                            if 'Ins_In' in grouped_res.columns and 'Tally_Sale_Count' in grouped_res.columns:
                                grouped_res['Ins_Pen_Pct'] = (grouped_res['Ins_In'] / grouped_res['Tally_Sale_Count'] * 100).fillna(0)

                            gt = grouped_res.sum(numeric_only=True)
                            
                            if 'Total_In_House' in gt and 'Total_Fin_InOut' in gt and gt['Total_Fin_InOut'] > 0:
                                gt['Fin_In_House_Pct'] = (gt['Total_In_House'] / gt['Total_Fin_InOut'] * 100)
                            else: gt['Fin_In_House_Pct'] = 0
                                
                            if 'MMFSL_Fin_In' in gt and 'Total_In_House' in gt and gt['Total_In_House'] > 0:
                                gt['MMFSL_Share_Pct'] = (gt['MMFSL_Fin_In'] / gt['Total_In_House'] * 100)
                            else: gt['MMFSL_Share_Pct'] = 0
                                
                            if 'Ins_In' in gt and 'Tally_Sale_Count' in gt and gt['Tally_Sale_Count'] > 0:
                                gt['Ins_Pen_Pct'] = (gt['Ins_In'] / gt['Tally_Sale_Count'] * 100)
                            else: gt['Ins_Pen_Pct'] = 0
                            
                            if len(grp_cols) > 1:
                                gt_name = tuple(['GRAND TOTAL'] + [''] * (len(grp_cols) - 1))
                            else:
                                gt_name = 'GRAND TOTAL'
                            
                            grouped_res.loc[gt_name] = gt

                            pivot_m = generate_month_wise_pivot(all_rep_df, grp_cols, start_date=ar_start, end_date=ar_end)
                            pivot_m = pivot_m.iloc[:-1]
                            pivot_m = pivot_m.drop(columns=['Total', 'Average'], errors='ignore')
                            
                            final_report = pd.merge(grouped_res, pivot_m, left_index=True, right_index=True, how='left')
                            month_cols = [c for c in pivot_m.columns]
                            if month_cols:
                                final_report.loc[gt_name, month_cols] = final_report[month_cols].sum(numeric_only=True)

                            col_map = {
                                'Other_Fin_In': 'Other Fin In-House',
                                'MMFSL_Fin_In': 'MMFSL Fin In-House',
                                'Total_In_House': 'Total IN HOUSE',
                                'Own_Lease': 'OWN FUNDS / LEASING',
                                'Total_Out_House': 'Total OUT HOUSE',
                                'Total_Fin_InOut': 'Total Fin (In+Out)',
                                'Finance Grand Total': 'Finance Grand Total',
                                'Tally_Sale_Count': 'Tally Sales Count',
                                'Difference': 'Difference',
                                'Fin_In_House_Pct': 'Fin In-House %',
                                'MMFSL_Share_Pct': 'MMFSL Share %',
                                'Ins_In': 'Total In-House (Ins)',
                                'Ins_Out': 'Total Out-House (Ins)',
                                'Insurance Grand Total': 'Insurance Grand Total',
                                'Ins_Pen_Pct': 'Ins In-House %'
                            }
                            final_report.rename(columns=col_map, inplace=True)
                            
                            desired_order = [
                                'Other Fin In-House', 'MMFSL Fin In-House', 'Total IN HOUSE',
                                'OWN FUNDS / LEASING', 'Total OUT HOUSE', 'Total Fin (In+Out)', 
                                'Finance Grand Total', 'Tally Sales Count', 'Difference',
                                'Fin In-House %', 'MMFSL Share %',
                                'Total In-House (Ins)', 'Total Out-House (Ins)', 'Insurance Grand Total', 'Ins In-House %'
                            ]
                            
                            final_cols = [c for c in desired_order if c in final_report.columns] + month_cols
                            return final_report[final_cols]

                        # --- DROPDOWN SELECTION ---
                        report_select = st.selectbox("Select Report Type:", [
                            "1. Consultant & Segment",
                            "2. ASM Performance",
                            "3. Model Wise Performance",
                            "4. Consultant Wise Sale Report",
                            "5. Consultant Consolidate Report",
                            "6. Consultant Consolidate Report (Model Wise)"
                        ])

                        if report_select == "1. Consultant & Segment":
                            st.markdown("#### 1. Consultant & Segment")
                            grp = [c for c in ["Sales Consultant Name", "ASM", "Sales Manager", "Segment"] if c in all_rep_df.columns]
                            if grp:
                                piv = generate_month_wise_pivot(all_rep_df, grp, start_date=ar_start, end_date=ar_end)
                                # Default Indian Number Format for counts
                                st.dataframe(piv.style.format(format_lakhs).format(subset=["Average"], formatter="{:.1f}"))
                        
                        elif report_select == "2. ASM Performance":
                            st.markdown("#### 2. ASM Performance")
                            asm_grp = [c for c in ["ASM", "Segment"] if c in all_rep_df.columns]
                            if asm_grp:
                                piv_asm = generate_month_wise_pivot(all_rep_df, asm_grp, start_date=ar_start, end_date=ar_end)
                                st.dataframe(piv_asm.style.format(format_lakhs).format(subset=["Average"], formatter="{:.1f}"))

                        elif report_select == "3. Model Wise Performance":
                            st.markdown("#### 3. Model Wise Performance")
                            model_grp = [c for c in ["Model", "Segment"] if c in all_rep_df.columns]
                            if model_grp:
                                piv_mod = generate_month_wise_pivot(all_rep_df, model_grp, start_date=ar_start, end_date=ar_end)
                                st.dataframe(piv_mod.style.format(format_lakhs).format(subset=["Average"], formatter="{:.1f}"))

                        elif report_select == "4. Consultant Wise Sale Report":
                            st.markdown("#### 4. Consultant Wise Sale Report")
                            if "Sales Consultant Name" in all_rep_df.columns:
                                cons_grp = ["Sales Consultant Name"]
                                piv_cons_only = generate_month_wise_pivot(all_rep_df, cons_grp, start_date=ar_start, end_date=ar_end)
                                st.dataframe(piv_cons_only.style.format(format_lakhs).format(subset=["Average"], formatter="{:.1f}"))

                        elif report_select == "5. Consultant Consolidate Report":
                            st.markdown("#### 5. Consultant Consolidate Report")
                            cons_consolidate_grp = [c for c in ["Sales Consultant Name", "Segment", "ASM", "Sales Manager", "SM", "Outlet"] if c in all_rep_df.columns]
                            if cons_consolidate_grp:
                                rep5 = create_consolidated_report(cons_consolidate_grp)
                                s5 = rep5.style.format(format_lakhs) # Default to indian numbers for counts
                                s5 = s5.format(subset=['Fin In-House %', 'MMFSL Share %', 'Ins In-House %'], formatter="{:.1f}")
                                st.dataframe(s5)

                        elif report_select == "6. Consultant Consolidate Report (Model Wise)":
                            st.markdown("#### 6. Consultant Consolidate Report (Model Wise)")
                            cons_consolidate_grp_model = [c for c in ["Sales Consultant Name", "Segment", "Model", "ASM", "Sales Manager", "SM", "Outlet"] if c in all_rep_df.columns]
                            if cons_consolidate_grp_model:
                                rep6 = create_consolidated_report(cons_consolidate_grp_model)
                                s6 = rep6.style.format(format_lakhs)
                                s6 = s6.format(subset=['Fin In-House %', 'MMFSL Share %', 'Ins In-House %'], formatter="{:.1f}")
                                st.dataframe(s6)

    if auto_refresh: time.sleep(refresh_rate); st.rerun()