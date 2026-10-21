import gradio as gr
import tensorflow as tf
import numpy as np
import json, os, re, time, secrets, hashlib
import html as H
from datetime import datetime, date

model = tf.keras.models.load_model("resnet50_final.keras")

USERS_FILE = "users.json"
HIST_FILE = "history.json"

def load_json(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

def esc(s):
    return H.escape(str(s))

def hash_pw(pw, salt_hex):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt_hex), 200000).hex()

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

:root, .dark, .gradio-container {
  --body-background-fill: #0B1220 !important;
  --background-fill-primary: #111C2E !important;
  --background-fill-secondary: #111C2E !important;
  --block-background-fill: #111C2E !important;
  --block-border-color: #263449 !important;
  --border-color-primary: #263449 !important;
  --input-background-fill: #0B1220 !important;
  --body-text-color: #F1F5F9 !important;
  --block-label-text-color: #94A3B8 !important;
}
body, .gradio-container { background: #0B1220 !important; color: #F1F5F9 !important;
                          font-family: 'Poppins', sans-serif !important; }

.topbar { display: flex; align-items: center; gap: 16px; background: #111C2E;
          border: 1px solid #263449; border-radius: 18px; padding: 18px 22px; margin: 18px 0 20px; }
.logo { width: 52px; height: 52px; border-radius: 14px; background: #14B8A6;
        display: flex; align-items: center; justify-content: center; font-size: 1.7rem; }
.brand { font-size: 1.6rem; font-weight: 700; color: #F1F5F9; line-height: 1.1; }
.tagline { color: #94A3B8; font-size: 0.95rem; }
.badge { margin-left: auto; background: rgba(20,184,166,0.15); color: #14B8A6;
         padding: 5px 16px; border-radius: 20px; font-size: 0.9rem; font-weight: 600; }

#login_box { max-width: 450px; margin: 50px auto 0; background: #111C2E !important;
             border: 1px solid #263449 !important; border-radius: 20px; padding: 26px !important; }
.login-head { text-align: center; margin-bottom: 8px; }
.login-head .logo { margin: 0 auto 12px; }
.login-head .brand { font-size: 1.8rem; }
button[role="tab"] { color: #94A3B8 !important; font-weight: 600 !important; }
button[role="tab"].selected { color: #14B8A6 !important; border-color: #14B8A6 !important; }
.err { color: #FCA5A5; background: rgba(229,57,53,0.12); border: 1px solid rgba(229,57,53,0.4);
       border-radius: 10px; padding: 8px 12px; text-align: center; font-size: 0.9rem; }
.ok { color: #86EFAC; background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.4);
      border-radius: 10px; padding: 8px 12px; text-align: center; font-size: 0.9rem; }

#analyze_btn, #clear_btn, #history_btn, #login_btn, #signup_btn, #logout_btn {
  border-radius: 12px !important; font-weight: 600 !important; min-height: 50px !important;
  transition: transform 0.15s ease, filter 0.15s ease, box-shadow 0.15s ease !important;
}
#analyze_btn:hover, #clear_btn:hover, #history_btn:hover, #login_btn:hover,
#signup_btn:hover, #logout_btn:hover { transform: translateY(-2px); filter: brightness(1.12); }
#analyze_btn, #login_btn, #signup_btn { background: #14B8A6 !important; color: #0B1220 !important;
                           border: 1px solid #14B8A6 !important; font-size: 1.05rem !important; }
#analyze_btn:hover, #login_btn:hover, #signup_btn:hover { box-shadow: 0 6px 18px rgba(20,184,166,0.35); }
#clear_btn, #logout_btn { background: #1E293B !important; color: #F1F5F9 !important; border: 1px solid #263449 !important; }
#history_btn { background: #172554 !important; color: #93C5FD !important; border: 1px solid #263449 !important; }
#logout_btn { min-height: 40px !important; margin-top: 22px; }

.card { border-radius: 18px; padding: 26px; text-align: center; color: white; }
.card h2 { margin: 0; font-size: 1.7rem; font-weight: 700; color: white; }
.card .patient { opacity: 0.95; font-size: 0.95rem; margin-bottom: 8px; }
.card .pct { font-size: 3.2rem; font-weight: 700; margin: 10px 0 4px; }
.card .sub { opacity: 0.9; margin-bottom: 14px; }
.card .note { margin-top: 14px; font-size: 0.9rem; opacity: 0.9; }
.bar { background: rgba(255,255,255,0.3); border-radius: 10px; height: 12px; overflow: hidden; }
.bar > div { background: white; height: 100%; border-radius: 10px; }
.red { background: linear-gradient(135deg, #e53935, #b71c1c); }
.amber { background: linear-gradient(135deg, #fbc02d, #ef6c00); }
.green { background: linear-gradient(135deg, #43a047, #1b5e20); }
.idle { background: #111C2E; border: 1px solid #263449; color: #F1F5F9; }
.idle h2 { color: #F1F5F9; font-size: 1.3rem; }

.hist { background: #111C2E; border: 1px solid #263449; border-radius: 14px;
        padding: 16px 18px; color: #F1F5F9; overflow-x: auto; }
.hist h3 { margin: 0 0 10px; color: #93C5FD; }
.htable { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
.htable th { text-align: left; color: #93C5FD; padding: 8px 10px; border-bottom: 1px solid #263449; }
.htable td { padding: 9px 10px; border-bottom: 1px solid #263449; }
.htable tr:last-child td { border-bottom: none; }
.tag { padding: 3px 12px; border-radius: 20px; color: white; font-size: 0.8rem; white-space: nowrap; }

/* per-record delete button + hidden helper components */
.delbtn { background: rgba(229,57,53,0.12); color: #FCA5A5; border: 1px solid rgba(229,57,53,0.4);
          border-radius: 8px; padding: 4px 12px; font-size: 0.8rem; font-weight: 600;
          cursor: pointer; white-space: nowrap; font-family: inherit; }
.delbtn:hover { filter: brightness(1.2); }
.htable td.act, .htable th.act { text-align: right; }
#del_id, #del_btn { display: none !important; }

.footer { text-align: center; color: #64748B; font-size: 0.8rem; margin: 26px 0 10px; }
</style>
"""

# Eye toggle for password fields (Login + Sign up password only) and
# the helper used by each history row's Delete button.
APP_JS = """
() => {
  window.pneumoDelete = (id) => {
    const t = document.querySelector('#del_id input, #del_id textarea');
    const b = document.querySelector('#del_btn');
    if (!t || !b) return;
    t.value = id;
    t.dispatchEvent(new Event('input', { bubbles: true }));
    setTimeout(() => b.click(), 150);
  };

  const EYE = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
  const EYE_OFF = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>';

  const attach = () => {
    ['l_pass', 's_pass'].forEach((id) => {
      const wrap = document.getElementById(id);
      if (!wrap) return;
      const input = wrap.querySelector('input');
      if (!input || wrap.querySelector('.eye-btn')) return;
      const holder = input.parentElement;
      holder.style.position = 'relative';
      input.style.paddingRight = '44px';
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'eye-btn';
      btn.title = 'Show / hide password';
      btn.innerHTML = EYE;
      btn.style.cssText = 'position:absolute;right:10px;bottom:6px;width:32px;height:32px;' +
        'display:flex;align-items:center;justify-content:center;background:transparent;' +
        'border:none;color:#94A3B8;cursor:pointer;padding:0;z-index:5;';
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        const show = input.type === 'password';
        input.type = show ? 'text' : 'password';
        btn.innerHTML = show ? EYE_OFF : EYE;
        btn.style.color = show ? '#14B8A6' : '#94A3B8';
      });
      holder.appendChild(btn);
    });
  };

  attach();
  new MutationObserver(attach).observe(document.body, { childList: true, subtree: true });
}
"""

LOGIN_HEAD = """
<div class='login-head'>
  <div class='logo'>🫁</div>
  <div class='brand'>PneumoAI</div>
  <div class='tagline'>Sign in or create an account</div>
</div>
"""

def header(name=""):
    badge = f"<span class='badge'>👤 {esc(name)}</span>" if name else ""
    return f"""
    <div class='topbar'>
      <div class='logo'>🫁</div>
      <div>
        <div class='brand'>PneumoAI</div>
        <div class='tagline'>AI-Powered Chest X-ray Analysis</div>
      </div>
      {badge}
    </div>
    """

FOOTER = "<div class='footer'>Decision-support tool only. Not a replacement for a medical professional.</div>"
IDLE = "<div class='card idle'><h2>Awaiting X-ray</h2><div>Fill the patient details, upload an image and click Analyse X-ray.</div></div>"

def notice(text):
    return f"<div class='card idle'><h2>{esc(text)}</h2></div>"

def err(t): return f"<div class='err'>{esc(t)}</div>"
def ok(t): return f"<div class='ok'>{esc(t)}</div>"

# ---------- SIGN UP ----------
def signup(full_name, username, pw, pw2):
    full_name = (full_name or "").strip()
    u = (username or "").strip().lower()
    if not full_name:
        return err("Please enter your full name."), gr.update(), gr.update(), gr.update(), gr.update()
    if not re.fullmatch(r"[a-z0-9_]{3,20}", u):
        return err("Username: 3-20 letters, numbers or underscore."), gr.update(), gr.update(), gr.update(), gr.update()
    if len(pw or "") < 8:
        return err("Password must be at least 8 characters."), gr.update(), gr.update(), gr.update(), gr.update()
    if pw != pw2:
        return err("Passwords do not match."), gr.update(), gr.update(), gr.update(), gr.update()
    users = load_json(USERS_FILE, {})
    if u in users:
        return err("That username is already taken."), gr.update(), gr.update(), gr.update(), gr.update()
    salt = secrets.token_hex(16)
    users[u] = {"name": full_name, "salt": salt, "hash": hash_pw(pw, salt)}
    save_json(USERS_FILE, users)
    return "", ok("Account created. Please log in."), gr.update(selected="login"), "", ""

# ---------- LOGIN ----------
def login(username, pw):
    u = (username or "").strip().lower()
    users = load_json(USERS_FILE, {})
    rec = users.get(u)
    if rec and secrets.compare_digest(hash_pw(pw or "", rec["salt"]), rec["hash"]):
        return (gr.update(visible=False), gr.update(visible=True), "", u, header(rec["name"]), "")
    time.sleep(1)
    return gr.update(), gr.update(), err("Invalid username or password."), "", gr.update(), gr.update()

def logout():
    return (gr.update(visible=True), gr.update(visible=False), "", "", "", gr.update(visible=False),
            False, IDLE, header(""), "",
            None, "", None, None, str(date.today()), "", "Newest first", "")

# ---------- HISTORY ----------
def render_history(search, sort, user):
    all_h = load_json(HIST_FILE, [])
    # give older records (saved before this update) a unique id so they can be deleted individually
    changed = False
    for r in all_h:
        if "id" not in r:
            r["id"] = secrets.token_hex(8)
            changed = True
    if changed:
        save_json(HIST_FILE, all_h)

    h = [r for r in all_h if r.get("user") == user]
    q = (search or "").strip().lower()
    if q:
        h = [r for r in h if q in r["name"].lower()]
    h.sort(key=lambda r: r["date"] + " " + r["time"], reverse=(sort != "Oldest first"))
    if not h:
        return "<div class='hist'><h3>Analysis History</h3>No records found.</div>"
    rows = "".join(
        f"<tr><td>{esc(r['name'])}</td><td>{r['age']}</td><td>{esc(r['gender'])}</td>"
        f"<td>{r['date']}</td><td>{r['time']}</td>"
        f"<td><span class='tag {r['cls']}'>{esc(r['result'])}</span></td><td>{r['pct']}%</td>"
        f"<td class='act'><button type='button' class='delbtn' "
        f"onclick=\"pneumoDelete('{esc(r['id'])}')\">Delete</button></td></tr>"
        for r in h
    )
    return ("<div class='hist'><h3>Analysis History</h3><table class='htable'>"
            "<tr><th>Patient</th><th>Age</th><th>Gender</th><th>Date</th><th>Time</th>"
            "<th>Result</th><th>Probability</th><th class='act'></th></tr>" + rows + "</table></div>")

def toggle_history(is_open, search, sort, user):
    new = not is_open
    return gr.update(visible=new), render_history(search, sort, user), new

def delete_record(rid, search, sort, user):
    rid = (rid or "").strip()
    if rid and user:
        all_h = load_json(HIST_FILE, [])
        all_h = [r for r in all_h if not (r.get("id") == rid and r.get("user") == user)]
        save_json(HIST_FILE, all_h)
    return render_history(search, sort, user)

# ---------- PREDICT ----------
def predict(img, name, age, gender, vdate, user, search, sort):
    if not user:
        return notice("Please log in first"), gr.update()
    if not name or not name.strip():
        return notice("Please enter the patient name"), gr.update()
    if age is None or age < 0 or age > 120:
        return notice("Please enter a valid age (0 to 120)"), gr.update()
    if not gender:
        return notice("Please select the gender"), gr.update()
    try:
        datetime.strptime(vdate.strip(), "%Y-%m-%d")
    except Exception:
        return notice("Date must look like 2026-10-21"), gr.update()
    if img is None:
        return notice("Please upload a chest X-ray"), gr.update()

    x = np.array(img.resize((224, 224)).convert("RGB"), dtype="float32")[None, ...]
    p = float(model.predict(x, verbose=0)[0][0])
    pct = p * 100

    if p >= 0.70:
        cls, title = "red", "Pneumonia Detected"
        note = "High likelihood of pneumonia. Clinical review recommended."
    elif p >= 0.40:
        cls, title = "amber", "Uncertain, Needs Review"
        note = "The model is not confident. Please have this X-ray checked by a doctor."
    else:
        cls, title = "green", "Normal"
        note = "No strong signs of pneumonia found."

    h = load_json(HIST_FILE, [])
    h.append({"id": secrets.token_hex(8), "user": user, "name": name.strip(), "age": int(age), "gender": gender,
              "date": vdate.strip(), "time": datetime.now().strftime("%H:%M:%S"),
              "result": title, "pct": round(pct, 1), "cls": cls})
    save_json(HIST_FILE, h)

    card = f"""
    <div class='card {cls}'>
      <div class='patient'>{esc(name.strip())} · {int(age)} yrs · {esc(gender)} · {esc(vdate.strip())}</div>
      <h2>{title}</h2>
      <div class='pct'>{pct:.1f}%</div>
      <div class='sub'>pneumonia probability</div>
      <div class='bar'><div style='width:{pct:.1f}%'></div></div>
      <div class='note'>{note}</div>
    </div>
    """
    return card, render_history(search, sort, user)

def clear_all():
    return None, IDLE, "", None, None, str(date.today())

# ---------- UI ----------
with gr.Blocks(title="PneumoAI") as demo:
    gr.HTML(STYLE)
    cur_user = gr.State("")
    hist_open = gr.State(False)

    # Login / Sign up page
    with gr.Column(elem_id="login_box") as login_col:
        gr.HTML(LOGIN_HEAD)
        with gr.Tabs() as tabs:
            with gr.Tab("Login", id="login"):
                l_user = gr.Textbox(label="Username", placeholder="Enter username")
                l_pass = gr.Textbox(label="Password", type="password", placeholder="Enter password",
                                    elem_id="l_pass")
                login_btn = gr.Button("Login", elem_id="login_btn")
                login_msg = gr.HTML("")
            with gr.Tab("Sign up", id="signup"):
                s_name = gr.Textbox(label="Full name", placeholder="e.g. Dr. Anita Kumar")
                s_user = gr.Textbox(label="Choose a username", placeholder="3-20 letters, numbers, _")
                s_pass = gr.Textbox(label="Password", type="password", placeholder="At least 8 characters",
                                    elem_id="s_pass")
                s_pass2 = gr.Textbox(label="Confirm password", type="password", placeholder="Re-enter password")
                signup_btn = gr.Button("Create account", elem_id="signup_btn")
                signup_msg = gr.HTML("")

    # Main app
    with gr.Column(visible=False) as app_col:
        with gr.Row():
            with gr.Column(scale=8):
                header_out = gr.HTML(header(""))
            with gr.Column(scale=1, min_width=110):
                logout_btn = gr.Button("Logout", elem_id="logout_btn", size="sm")

        with gr.Row():
            with gr.Column():
                name_in = gr.Textbox(label="Patient name", placeholder="e.g. Anita Kumar")
                with gr.Row():
                    age_in = gr.Number(label="Age", precision=0, minimum=0, maximum=120)
                    gender_in = gr.Dropdown(["Female", "Male", "Other"], label="Gender")
                date_in = gr.Textbox(label="Date (YYYY-MM-DD)", value=str(date.today()))
                inp = gr.Image(type="pil", label="Upload chest X-ray")
                analyze_btn = gr.Button("🔍  Analyse X-ray", elem_id="analyze_btn")
                with gr.Row():
                    clear_btn = gr.Button("↺  Clear", elem_id="clear_btn")
                    history_btn = gr.Button("🕘  History", elem_id="history_btn")
            with gr.Column():
                out = gr.HTML(IDLE)

        with gr.Column(visible=False) as hist_panel:
            with gr.Row():
                search_in = gr.Textbox(label="Search by patient name", placeholder="Type a name...", scale=3)
                sort_in = gr.Dropdown(["Newest first", "Oldest first"], value="Newest first",
                                      label="Sort by date", scale=1)
            hist_out = gr.HTML("")

        # hidden helpers used by the per-row Delete buttons
        del_id = gr.Textbox(elem_id="del_id", label="del")
        del_btn = gr.Button("del", elem_id="del_btn")

        gr.HTML(FOOTER)

    # Events
    signup_btn.click(signup, [s_name, s_user, s_pass, s_pass2],
                     [signup_msg, login_msg, tabs, s_pass, s_pass2])
    login_btn.click(login, [l_user, l_pass], [login_col, app_col, login_msg, cur_user, header_out, l_pass])
    l_pass.submit(login, [l_user, l_pass], [login_col, app_col, login_msg, cur_user, header_out, l_pass])
    logout_btn.click(logout, None, [login_col, app_col, l_user, l_pass, login_msg, hist_panel,
                                    hist_open, out, header_out, cur_user,
                                    inp, name_in, age_in, gender_in, date_in,
                                    search_in, sort_in, hist_out])

    analyze_btn.click(predict, [inp, name_in, age_in, gender_in, date_in, cur_user, search_in, sort_in],
                      [out, hist_out])
    clear_btn.click(clear_all, None, [inp, out, name_in, age_in, gender_in, date_in])
    history_btn.click(toggle_history, [hist_open, search_in, sort_in, cur_user],
                      [hist_panel, hist_out, hist_open])
    search_in.change(render_history, [search_in, sort_in, cur_user], hist_out)
    sort_in.change(render_history, [search_in, sort_in, cur_user], hist_out)
    del_btn.click(delete_record, [del_id, search_in, sort_in, cur_user], hist_out)

demo.launch(js=APP_JS, share=True)