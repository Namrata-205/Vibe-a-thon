const buttons = document.querySelectorAll('.lang-btn');

// Restore previous selection
const savedLang = localStorage.getItem('selectedLang');
if (savedLang) {
  buttons.forEach(btn => {
    if (btn.dataset.lang === savedLang) {
      btn.classList.add('selected');
    }
  });
}

// Add event listeners
buttons.forEach(button => {
  button.addEventListener('click', () => {
    buttons.forEach(btn => btn.classList.remove('selected'));
    button.classList.add('selected');
    const selected = button.dataset.lang;
    localStorage.setItem('selectedLang', selected);
    console.log("Language set to:", selected);
  });
});
