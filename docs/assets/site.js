"use strict";

const isChinese = document.documentElement.lang === "zh-CN";
const status = document.getElementById("copy-status");
let statusTimer;

document.querySelectorAll("[data-language-link]").forEach((link) => {
  link.addEventListener("click", () => {
    const url = new URL(link.href);
    url.hash = location.hash;
    link.href = url.href;
  });
});

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    const code = document.getElementById(button.dataset.copy);
    button.disabled = true;
    clearTimeout(statusTimer);
    try {
      if (!navigator.clipboard) throw new Error("Clipboard API is unavailable.");
      await navigator.clipboard.writeText(code.textContent);
      status.textContent = isChinese ? "已复制到剪贴板。" : "Copied to clipboard.";
    } catch (error) {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(code);
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = isChinese
        ? "无法访问剪贴板。已选中文本，请手动复制。"
        : "Clipboard unavailable. Text selected; please copy manually.";
      console.warn("Could not copy to clipboard.", error);
    } finally {
      button.disabled = false;
    }
    statusTimer = setTimeout(() => { status.textContent = ""; }, 5000);
  });
});
