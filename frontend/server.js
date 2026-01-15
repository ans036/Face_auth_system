const express = require("express");
const app = express();

// Allow all JavaScript execution (no CSP restrictions)
app.use((req, res, next) => {
    res.setHeader("Content-Security-Policy", "default-src * 'unsafe-inline' 'unsafe-eval' data: blob:; script-src * 'unsafe-inline' 'unsafe-eval'; connect-src *;");
    next();
});

app.use(express.static("public"));
app.listen(3000, () => console.log("Frontend at http://localhost:3000"));
