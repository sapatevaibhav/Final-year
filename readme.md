clone the repo

create python virtual environment

launch the environment

install requirenments from txt file

`streamlit run app.py`





Below are detailed steps to deploy your Streamlit app on an Amazon EC2 instance and map it to your subdomain (detection.sapatevaibhav.me):

---

## 1. **Launch and Prepare an EC2 Instance**

1. **Launch an EC2 Instance:**
   - Log in to your AWS Management Console.
   - Navigate to EC2 and click “Launch Instance.”
   - Choose an appropriate AMI (for example, Ubuntu Server 20.04 LTS).
   - Select an instance type (e.g., t2.micro if eligible for free tier).
   - Configure instance details and ensure the security group allows HTTP (port 80) and HTTPS (port 443) along with SSH (port 22) for management.
   - Launch and download your key pair.

2. **SSH into Your Instance:**

   ```bash
   ssh -i /path/to/your-key.pem ubuntu@your-ec2-public-ip
   ```

3. **Update the Package List:**

   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

---

## 2. **Install Dependencies and Set Up the Environment**

1. **Install Python, Pip, and Virtual Environment:**

   ```bash
   sudo apt install python3-pip python3-venv -y
   ```

2. **Clone Your App Repository (or Transfer Your Code):**

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

3. **Create and Activate a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Required Packages:**

   Ensure your `requirements.txt` includes:
   - streamlit
   - opencv-python
   - imutils
   - tensorflow
   - pillow
   - any other libraries (e.g., numpy, pandas)

   Then install via:

   ```bash
   pip install -r requirements.txt
   ```

---

## 3. **Run Your Streamlit App**

1. **Test Locally on the Instance:**

   In your project directory, run:

   ```bash
   streamlit run app.py --server.enableCORS false --server.port 8501
   ```

   This starts the app on port 8501. You can test by opening `http://your-ec2-public-ip:8501` in a browser.

2. **(Optional) Use a Process Manager:**

   To keep your app running, consider using a process manager like `pm2` or `nohup` (or a systemd service). For example:

   ```bash
   nohup streamlit run app.py --server.enableCORS false --server.port 8501 &
   ```

---

## 4. **Set Up Nginx as a Reverse Proxy**

1. **Install Nginx:**

   ```bash
   sudo apt install nginx -y
   ```

2. **Configure Nginx:**

   Create a new configuration file (e.g., `/etc/nginx/sites-available/detection`):

   ```nginx
   server {
       listen 80;
       server_name detection.sapatevaibhav.me;

       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
       }
   }
   ```

3. **Enable the Nginx Site and Test Configuration:**

   ```bash
   sudo ln -s /etc/nginx/sites-available/detection /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

   Now, your Streamlit app will be available via `http://detection.sapatevaibhav.me`.

---

## 5. **Configure DNS for Your Subdomain**

1. **Log into Your DNS Provider:**
   - Go to your DNS management console.

2. **Create an A Record:**
   - **Name:** detection (or detection.sapatevaibhav.me if required)
   - **Type:** A
   - **Value:** Your EC2 instance’s public IP address

3. **Wait for DNS Propagation:**
   - This may take a few minutes up to an hour. Once propagated, accessing `http://detection.sapatevaibhav.me` should load your app.

---

## 6. **(Optional) Enable HTTPS with Let’s Encrypt**

For a secure connection, consider setting up SSL using Certbot:

1. **Install Certbot:**

   ```bash
   sudo apt install certbot python3-certbot-nginx -y
   ```

2. **Obtain and Install the SSL Certificate:**

   ```bash
   sudo certbot --nginx -d detection.sapatevaibhav.me
   ```

3. **Follow the Prompts:**
   - Certbot will automatically update your Nginx configuration to redirect HTTP to HTTPS.

---

## Final Notes

- **Security:**
  Ensure your EC2 security group is set to allow only the necessary ports (22 for SSH, 80 for HTTP, and 443 for HTTPS).

- **Monitoring & Maintenance:**
  Consider setting up monitoring/logging for your EC2 instance and renewing your SSL certificates automatically.

Following these steps, your Streamlit app will be running on an EC2 instance and available at the subdomain detection.sapatevaibhav.me. If you encounter any issues or need further clarification on any step, feel free to ask!
