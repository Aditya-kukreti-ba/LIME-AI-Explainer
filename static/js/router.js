"use strict";
/* ── Page Router — fluid transitions between pages ── */

const PAGE_ORDER = ['hero', 'text', 'tabular', 'image'];
let currentPage  = 'hero';

function navigateTo(target) {
  if (target === currentPage) return;

  const fromEl  = document.getElementById('page-' + currentPage);
  const toEl    = document.getElementById('page-' + target);
  const nav     = document.getElementById('app-nav');
  const fromIdx = PAGE_ORDER.indexOf(currentPage);
  const toIdx   = PAGE_ORDER.indexOf(target);

  // Decide exit / entry directions
  let exitState, entryState;
  if (currentPage === 'hero') {
    exitState  = 'above';
    entryState = 'below';
  } else if (target === 'hero') {
    exitState  = 'below';
    entryState = 'above';
  } else if (toIdx > fromIdx) {
    exitState  = 'left';
    entryState = 'right';
  } else {
    exitState  = 'right';
    entryState = 'left';
  }

  // Snap incoming page to its start position without animating
  toEl.style.transition = 'none';
  toEl.dataset.state    = entryState;
  toEl.getBoundingClientRect();          // force reflow
  toEl.style.transition = '';

  // Fire both transitions together
  fromEl.dataset.state = exitState;
  toEl.dataset.state   = 'active';

  // Nav visibility + active tab
  if (target === 'hero') {
    nav.classList.remove('nav-visible');
  } else {
    nav.classList.add('nav-visible');
    document.querySelectorAll('.nav-tab').forEach(t => {
      t.classList.toggle('active', t.dataset.page === target);
    });
  }

  // Reset scroll on incoming page
  const body = toEl.querySelector('.page-body');
  if (body) body.scrollTop = 0;

  currentPage = target;
}

/* ── Wire up all navigation triggers ── */

// Hero CTA
document.getElementById('hero-cta-btn')
  .addEventListener('click', () => navigateTo('text'));

// Hero scroll hint
document.getElementById('hero-scroll-btn')
  .addEventListener('click', () => navigateTo('text'));

// Hero feature pills
document.querySelectorAll('.hero-pill[data-page]').forEach(pill => {
  pill.addEventListener('click', () => navigateTo(pill.dataset.page));
});

// App nav tabs
document.querySelectorAll('.nav-tab').forEach(tab => {
  tab.addEventListener('click', () => navigateTo(tab.dataset.page));
});

// Home button
document.getElementById('nav-home-btn')
  .addEventListener('click', () => navigateTo('hero'));
