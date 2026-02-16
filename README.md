<!-- README.html (you can paste this into README.md too, GitHub will render the HTML) -->

<div align="center">
  <h1>🛡️ Hostel Security Assistance</h1>
  <p>
    A real-time <b>AI-powered hostel gate security system</b> that detects people, recognizes faces,
    and alerts operators through a simple dashboard.
  </p>

  <p>
    <a href="#-what-problem-does-this-solve">Problem</a> •
    <a href="#-what-it-does">What it does</a> •
    <a href="#-how-it-works">How it works</a> •
    <a href="#-dashboard-streamlit-web-app">Dashboard</a> •
    <a href="#-tech-stack">Tech</a> •
    <a href="#-quick-start">Quick start</a>
  </p>

  <hr style="width: 70%;"/>
</div>

<h2 id="-what-problem-does-this-solve">🎯 What Problem Does This Solve?</h2>
<ul>
  <li>Manual gate checks can miss unauthorized entries.</li>
  <li>Tailgating (multiple people entering together) is hard to detect reliably.</li>
  <li>Tracking who entered and when becomes messy without consistent logs.</li>
</ul>
<p>
  This system automates <b>face recognition</b>, <b>entry/exit logging</b>, <b>unknown-person alerts</b>,
  and <b>temporary guest access</b>.
</p>

<h2 id="-what-it-does">✅ What It Does</h2>
<table>
  <thead>
    <tr>
      <th align="left">Component</th>
      <th align="left">What it does</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Camera Worker (Backend Engine)</b></td>
      <td>Monitors live camera feeds, detects people, recognizes faces, and creates alerts/logs.</td>
    </tr>
    <tr>
      <td><b>Operations Dashboard (Streamlit Web App)</b></td>
      <td>Lets security staff review alerts, mark actions taken, and manage guest access.</td>
    </tr>
  </tbody>
</table>

<hr/>

<h2 id="-how-it-works">🔄 How It Works</h2>

<h3>1) System Startup</h3>
<ul>
  <li>Connects to entry and exit camera streams.</li>
  <li>Initializes the SQLite database schema.</li>
  <li>Starts two worker threads (entry + exit) and keeps running until stopped.</li>
</ul>

<h3>2) Live Camera Monitoring</h3>
<ul>
  <li>Detects <b>people</b> using YOLO.</li>
  <li>Runs person detection every <code>yolo_every_n_frames</code> frames for speed.</li>
  <li>Uses a “safe zone” filter (<code>min_inside_ratio</code>) to reduce border/edge false alerts.</li>
  <li>Runs face recognition only when a person is present.</li>
  <li>Auto-reconnects if the camera stream drops.</li>
</ul>

<h3>3) Face Recognition Engine</h3>
<ul>
  <li>Face detection + alignment: <b>MTCNN</b></li>
  <li>Face embeddings: <b>FaceNet (InceptionResnetV1, pretrained on vggface2)</b></li>
  <li>L2-normalized embeddings for stable similarity comparison.</li>
  <li>Identity database sources:
    <ul>
      <li><b>Permanent residents</b> from <code>data/known/</code></li>
      <li><b>Temporary guests</b> loaded from DB (only if not expired)</li>
    </ul>
  </li>
  <li>Open-set decision logic:
    <ul>
      <li>Dynamic thresholds based on number of identities</li>
      <li>Optional “best-vs-second” margin check</li>
      <li>Quality gates (min face probability + min face size)</li>
    </ul>
  </li>
</ul>

<h3>4) Smart Alert Logic (Reduces False Alarms)</h3>
<ul>
  <li>
    <b>Tailgating / mismatch detection:</b> if number of people ≠ number of valid faces for a minimum duration,
    raises an alert (rate-limited).
  </li>
  <li>
    <b>Identity smoothing (K-of-N voting):</b> avoids one-frame mistakes before confirming identity changes.
  </li>
  <li>
    <b>Unknown smoothing:</b> unknown faces must persist across time before an alert triggers (with cooldown).
  </li>
</ul>

<h3>5) Data Storage</h3>
<p>
  The system stores events in <b>SQLite</b> and saves evidence images to disk.
</p>
<p><b>Core tables:</b></p>
<ul>
  <li><code>alerts</code></li>
  <li><code>entry_decisions</code></li>
  <li><code>text_logs</code></li>
  <li><code>attendance_state</code></li>
  <li><code>guest_access</code></li>
  <li><code>events</code></li>
</ul>

<hr/>

<h2 id="-dashboard-streamlit-web-app">🖥️ Dashboard (Streamlit Web App)</h2>
<ul>
  <li>Secure login (bcrypt password check + brute-force lockout).</li>
  <li>Auto-refresh every ~2 seconds.</li>
  <li>View open alerts with evidence image preview.</li>
  <li>Mark alerts as:
    <ul>
      <li><b>Ignored</b> (false alert)</li>
      <li><b>Dealt</b> (handled) + optional extra details</li>
    </ul>
  </li>
  <li>Temporary guest face upload with expiry + worker reload flag.</li>
</ul>

<hr/>

<h2 id="-tech-stack">🛠️ Tech Stack</h2>

<h3>Main Libraries</h3>
<ul>
  <li><b>CV/ML:</b> <code>opencv-python</code>, <code>ultralytics</code>, <code>facenet-pytorch</code>, <code>torch</code></li>
  <li><b>UI:</b> <code>streamlit</code>, <code>streamlit-autorefresh</code></li>
  <li><b>Security:</b> <code>bcrypt</code></li>
  <li><b>Database:</b> <code>sqlite3</code> (built-in Python)</li>
</ul>

<hr/>

<h2 id="-quick-start">🚀 Quick Start</h2>

<div style="border:1px solid #ddd; border-radius:10px; padding:12px; background:#fafafa;">
  <p style="margin-top:0;">
    <b>Important:</b> Add at least <b>2 known identities</b> in <code>data/known/</code> before running.
    Each identity should be a folder named after the person, containing multiple clear face images.
  </p>

  <pre><code>data/known/
  ├── Rahul/
  │   ├── img1.jpg
  │   └── img2.jpg
  └── Amit/
      ├── img1.jpg
      └── img2.jpg</code></pre>
</div>

<h3>1) Clone the repository</h3>
<pre><code>git clone &lt;repo_url&gt;
cd hostel-security-assistance</code></pre>

<h3>2) Install dependencies</h3>
<pre><code>pip install -r requirements.txt</code></pre>

<h3>3) Start camera workers</h3>
<pre><code>python run_workers.py</code></pre>

<h3>4) Start the dashboard (ID passwords are in activation.txt file)</h3>
<pre><code>streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
</code></pre>

<hr/>

<h2>✨ Key Features</h2>
<ul>
  <li>Real-time face recognition</li>
  <li>Tailgating detection</li>
  <li>Unknown-person alerts</li>
  <li>Temporary guest access with expiry</li>
  <li>Evidence image storage</li>
  <li>Secure operator login</li>
  <li>Lightweight SQLite database</li>
  <li>Automatic camera reconnect</li>
</ul>

<h2>📌 Ideal Use Cases</h2>
<ul>
  <li>Hostel gates</li>
  <li>PG accommodations</li>
  <li>Private campuses</li>
  <li>Small institutions</li>
</ul>

<h2>⚠️ Notes</h2>
<ul>
  <li>Camera quality and lighting strongly affect accuracy.</li>
  <li>More images per person generally improves recognition reliability.</li>
  <li>Using a GPU will significantly improve performance (if available).</li>
</ul>
