
  (() => {
    document.querySelectorAll('.standalone-results-table').forEach((block) => {
      const pages = Array.from(block.querySelectorAll('.standalone-table-page'));
      const tabs = Array.from(block.querySelectorAll('.table-switcher-tab'));
      let index = 0;
      let direction = 'right';
      const render = () => {
        pages.forEach((page, i) => {
          page.hidden = i !== index;
          page.classList.remove('switcher-slide-left', 'switcher-slide-right');
          if (i === index) {
            page.classList.add('switcher-panel');
            page.classList.add('switcher-slide-' + direction);
          }
        });
        tabs.forEach((tab, i) => tab.classList.toggle('active', i === index));
      };
      block.querySelectorAll('[data-table-target]').forEach((button) => {
        button.addEventListener('click', () => {
          const newIdx = Number(button.getAttribute('data-table-target') || 0);
          direction = newIdx > index ? 'right' : 'left';
          index = newIdx;
          render();
        });
      });
      block.querySelectorAll('[data-table-nav]').forEach((button) => {
        button.addEventListener('click', () => {
          const delta = Number(button.getAttribute('data-table-nav') || 0);
          direction = delta > 0 ? 'right' : 'left';
          index = (index + delta + pages.length) % pages.length;
          render();
        });
      });
      render();
    });

    document.querySelectorAll('.standalone-carousel').forEach((carousel) => {
      const slides = Array.from(carousel.querySelectorAll('.standalone-carousel-slide'));
      const indicator = carousel.querySelector('.carousel-indicator');
      let index = 0;
      const render = () => {
        slides.forEach((slide, i) => { slide.hidden = i !== index; });
        if (indicator) indicator.textContent = (index + 1) + ' / ' + slides.length;
      };
      carousel.querySelectorAll('[data-carousel-nav]').forEach((button) => {
        button.addEventListener('click', () => {
          const delta = Number(button.getAttribute('data-carousel-nav') || 0);
          index = (index + delta + slides.length) % slides.length;
          render();
        });
      });
      render();
    });

    document.querySelectorAll('.comparison-container').forEach((container) => {
      const overlay = container.querySelector('.comparison-overlay');
      const line = container.querySelector('.comparison-slider-line');
      if (!overlay || !line) return;
      let dragging = false;
      const setPosition = (clientX) => {
        const rect = container.getBoundingClientRect();
        const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
        const pct = (x / rect.width) * 100;
        overlay.style.clipPath = 'inset(0 ' + (100 - pct) + '% 0 0)';
        line.style.left = pct + '%';
      };
      const onMove = (event) => {
        if (!dragging) return;
        const point = event.touches ? event.touches[0] : event;
        setPosition(point.clientX);
      };
      const onEnd = () => { dragging = false; };
      line.addEventListener('mousedown', () => { dragging = true; });
      line.addEventListener('touchstart', () => { dragging = true; }, { passive: true });
      window.addEventListener('mousemove', onMove);
      window.addEventListener('touchmove', onMove, { passive: true });
      window.addEventListener('mouseup', onEnd);
      window.addEventListener('touchend', onEnd);
      container.addEventListener('click', (event) => setPosition(event.clientX));
    });

    // Chart tab switching
    document.querySelectorAll('.standalone-chart-tabs').forEach((block) => {
      const pages = Array.from(block.querySelectorAll('.standalone-chart-page'));
      const tabs = Array.from(block.querySelectorAll('[data-chart-target]'));
      const navBtns = Array.from(block.querySelectorAll('.table-switcher-nav'));
      let index = 0;
      let direction = 'right';
      const render = () => {
        pages.forEach((page, i) => {
          page.style.display = i === index ? '' : 'none';
          page.classList.remove('switcher-slide-left', 'switcher-slide-right');
          if (i === index) {
            page.classList.add('switcher-panel');
            page.classList.add('switcher-slide-' + direction);
          }
        });
        tabs.forEach((tab, i) => tab.classList.toggle('active', i === index));
      };
      tabs.forEach((tab) => {
        tab.addEventListener('click', () => {
          const newIdx = Number(tab.getAttribute('data-chart-target') || 0);
          direction = newIdx > index ? 'right' : 'left';
          index = newIdx;
          render();
        });
      });
      if (navBtns.length >= 2) {
        navBtns[0].addEventListener('click', () => { direction = 'left'; index = (index - 1 + pages.length) % pages.length; render(); });
        navBtns[1].addEventListener('click', () => { direction = 'right'; index = (index + 1) % pages.length; render(); });
      }
      render();
    });

    // Sidebar nav: scroll spy + smooth scroll
    document.querySelectorAll('.sec-nav-sidebar').forEach((nav) => {
      const links = Array.from(nav.querySelectorAll('.sidebar-nav-item'));
      const ids = links.map(a => a.getAttribute('href')?.replace('#', '')).filter(Boolean);
      const targets = ids.map(id => document.getElementById(id)).filter(Boolean);
      if (!targets.length) return;

      // Read accent color: resolve CSS variables to actual computed colors
      const activeLink = nav.querySelector('.sidebar-nav-item.active');
      const accentColor = (activeLink ? getComputedStyle(activeLink).color : '') ||
        getComputedStyle(document.documentElement).getPropertyValue('--primary').trim() ||
        '';
      // Read default (inactive) text color
      const inactiveLink = nav.querySelector('.sidebar-nav-item:not(.active)');
      const defaultColor = (inactiveLink ? getComputedStyle(inactiveLink).color : '') || '';
      // Read inactive dot color from first inactive dot
      const inactiveDot = nav.querySelector('.sidebar-nav-item:not(.active) .sidebar-nav-dot');
      const dotDefaultBg = inactiveDot ? getComputedStyle(inactiveDot).backgroundColor : '';
      let activeIdx = 0;

      const setActive = (idx) => {
        activeIdx = idx;
        links.forEach((link, i) => {
          const isActive = i === idx;
          link.classList.toggle('active', isActive);
          link.style.fontWeight = isActive ? '700' : '400';
          link.style.color = isActive ? accentColor : defaultColor;
          // Update dot background
          const dot = link.querySelector('.sidebar-nav-dot');
          if (dot) {
            dot.style.background = isActive ? accentColor : dotDefaultBg;
            dot.style.transform = isActive ? 'scale(1.3)' : 'scale(1)';
          }
        });
        // Update dots track indicator
        const track = nav.querySelector('[data-track-indicator]');
        if (track) track.style.top = (idx * 38) + 'px';
      };

      // Intersection observer for scroll spy
      const observer = new IntersectionObserver((entries) => {
        let topIdx = -1, topY = Infinity;
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const idx = targets.indexOf(entry.target);
            if (idx >= 0 && entry.boundingClientRect.top < topY) {
              topY = entry.boundingClientRect.top;
              topIdx = idx;
            }
          }
        });
        if (topIdx >= 0) setActive(topIdx);
      }, { rootMargin: '-10% 0px -60% 0px', threshold: 0 });

      targets.forEach(el => observer.observe(el));

      // Smooth scroll on click
      links.forEach((link, i) => {
        link.addEventListener('click', (e) => {
          e.preventDefault();
          setActive(i);
          targets[i]?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
      });
    });
  })();
  