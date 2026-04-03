// Intersection Observer — trigger animations when sections enter view
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('in-view');
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.15 });

document.querySelectorAll('.result-card, .pipe-step, .cluster-card, .concl-block, .scard').forEach(el => {
  observer.observe(el);
});

// Active nav link on scroll
const sections = document.querySelectorAll('section[id]');
const navLinks  = document.querySelectorAll('.nav-links a');

const sectionObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const id = entry.target.getAttribute('id');
      navLinks.forEach(a => {
        a.style.color = a.getAttribute('href') === `#${id}`
          ? 'var(--green)'
          : '';
      });
    }
  });
}, { threshold: 0.4 });

sections.forEach(s => sectionObserver.observe(s));

// RPD fill bars — re-trigger on scroll into view
const rpdFills = document.querySelectorAll('.rpd-fill, .dt-fill, .rc-r2-fill');
const fillObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.animationPlayState = 'running';
      fillObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.5 });

rpdFills.forEach(el => {
  el.style.animationPlayState = 'paused';
  fillObserver.observe(el);
});
