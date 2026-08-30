function copyBibTeX() {
  const bibtex = document.getElementById('bibtex-code');
  const button = document.querySelector('.copy-bibtex-btn');
  const label = button?.querySelector('.copy-text');

  if (!bibtex || !button || !label) return;

  navigator.clipboard.writeText(bibtex.textContent || '').then(() => {
    button.classList.add('copied');
    label.textContent = 'Copied';
    window.setTimeout(() => {
      button.classList.remove('copied');
      label.textContent = 'Copy';
    }, 1800);
  });
}

function updateScrollButton() {
  const button = document.querySelector('.scroll-to-top');
  if (!button) return;
  button.classList.toggle('visible', window.scrollY > 360);
}

function setupOfflineReleaseDialog() {
  const trigger = document.querySelector('[data-offline-release]');
  const modal = document.getElementById('offline-release-modal');
  const closeButtons = modal?.querySelectorAll('[data-release-close]');
  const continueLink = modal?.querySelector('[data-release-continue]');
  const copyButton = modal?.querySelector('[data-copy-release-code]');
  const closeButton = modal?.querySelector('.release-modal-close');
  let previousFocus = null;

  if (!trigger || !modal) return;

  const closeModal = () => {
    modal.hidden = true;
    document.body.classList.remove('modal-open');
    previousFocus?.focus();
  };

  trigger.addEventListener('click', (event) => {
    event.preventDefault();
    previousFocus = document.activeElement;
    modal.hidden = false;
    document.body.classList.add('modal-open');
    closeButton?.focus();
  });

  closeButtons?.forEach((button) => button.addEventListener('click', closeModal));
  continueLink?.addEventListener('click', closeModal);
  copyButton?.addEventListener('click', () => {
    navigator.clipboard.writeText('sjtu').then(() => {
      copyButton.textContent = 'Copied';
      window.setTimeout(() => {
        copyButton.textContent = 'Copy';
      }, 1600);
    });
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !modal.hidden) closeModal();
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const copyButton = document.querySelector('.copy-bibtex-btn');
  const scrollButton = document.querySelector('.scroll-to-top');

  copyButton?.addEventListener('click', copyBibTeX);
  setupOfflineReleaseDialog();
  scrollButton?.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  updateScrollButton();
  window.addEventListener('scroll', updateScrollButton, { passive: true });
});
