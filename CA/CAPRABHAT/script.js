// ─── PRELOADER ────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
    const preloader = document.getElementById('preloader');
    setTimeout(() => {
        preloader.classList.add('hidden');
        startCounters();
        startTyping();
    }, 800);
});

// ─── HERO PARTICLES ───────────────────────────────────────────────────────────
(function createParticles() {
    const container = document.getElementById('heroParticles');
    if (!container) return;
    for (let i = 0; i < 25; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        p.style.cssText = `
            left: ${Math.random() * 100}%;
            width: ${Math.random() * 4 + 2}px;
            height: ${Math.random() * 4 + 2}px;
            animation-duration: ${Math.random() * 15 + 8}s;
            animation-delay: ${Math.random() * 10}s;
        `;
        container.appendChild(p);
    }
})();

// ─── TYPEWRITER ───────────────────────────────────────────────────────────────
const phrases = [
    'Income Tax Notices',
    'Tax Notice Resolution',
    'Notice Defence',
    'ITR Filing',
    'Notice Defence',
    'Business Advisory'
];

let phraseIndex = 0, charIndex = 0, isDeleting = false;

function startTyping() {
    const el = document.getElementById('typedText');
    if (!el) return;
    typeWriter(el);
}

function typeWriter(el) {
    const current = phrases[phraseIndex];
    if (isDeleting) {
        el.textContent = current.slice(0, charIndex--);
    } else {
        el.textContent = current.slice(0, charIndex++);
    }

    let delay = isDeleting ? 60 : 100;

    if (!isDeleting && charIndex === current.length + 1) {
        delay = 2000;
        isDeleting = true;
    } else if (isDeleting && charIndex === -1) {
        isDeleting = false;
        phraseIndex = (phraseIndex + 1) % phrases.length;
        charIndex = 0;
        delay = 400;
    }

    setTimeout(() => typeWriter(el), delay);
}

// ─── STAT COUNTERS ────────────────────────────────────────────────────────────
function startCounters() {
    document.querySelectorAll('.stat-number').forEach(el => {
        const target = parseInt(el.dataset.target);
        const duration = 1800;
        const step = target / (duration / 16);
        let current = 0;

        const update = () => {
            current = Math.min(current + step, target);
            el.textContent = Math.floor(current);
            if (current < target) requestAnimationFrame(update);
            else el.textContent = target;
        };
        requestAnimationFrame(update);
    });
}

// ─── NAVBAR SCROLL EFFECT ─────────────────────────────────────────────────────
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
    if (window.pageYOffset > 50) navbar.classList.add('scrolled');
    else navbar.classList.remove('scrolled');
}, { passive: true });

// ─── HAMBURGER MENU ───────────────────────────────────────────────────────────
const hamburger = document.getElementById('hamburger');
const navLinks = document.getElementById('navLinks');

hamburger.addEventListener('click', () => {
    hamburger.classList.toggle('open');
    navLinks.classList.toggle('open');
});

// Close on link click
navLinks.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
        hamburger.classList.remove('open');
        navLinks.classList.remove('open');
    });
});

// Close on outside click
document.addEventListener('click', e => {
    if (!navbar.contains(e.target)) {
        hamburger.classList.remove('open');
        navLinks.classList.remove('open');
    }
});

// ─── SMOOTH SCROLL + ACTIVE NAV ───────────────────────────────────────────────
function scrollToSection(id) {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: 'smooth' });
}

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            e.preventDefault();
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

window.addEventListener('scroll', () => {
    const sections = document.querySelectorAll('section[id]');
    const links = document.querySelectorAll('.nav-links a');
    let current = '';

    sections.forEach(section => {
        if (window.pageYOffset >= section.offsetTop - 120) {
            current = section.id;
        }
    });

    links.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === '#' + current) {
            link.classList.add('active');
        }
    });
}, { passive: true });

// ─── FADE-IN ON SCROLL ────────────────────────────────────────────────────────
const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry, i) => {
        if (entry.isIntersecting) {
            setTimeout(() => entry.target.classList.add('visible'), i * 80);
            observer.unobserve(entry.target);
        }
    });
}, { threshold: 0.1, rootMargin: '0px 0px -60px 0px' });

document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));

// ─── SCROLL TO TOP ────────────────────────────────────────────────────────────
const scrollTopBtn = document.getElementById('scrollTop');

window.addEventListener('scroll', () => {
    if (window.pageYOffset > 400) scrollTopBtn.classList.add('visible');
    else scrollTopBtn.classList.remove('visible');
}, { passive: true });

scrollTopBtn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
});

// ─── TESTIMONIALS SLIDER ──────────────────────────────────────────────────────
let currentSlide = 0;
const cards = document.querySelectorAll('.testimonial-card');
const dots = document.querySelectorAll('.dot');
let autoSlide = null;

function showSlide(index) {
    cards.forEach(c => c.classList.remove('active'));
    dots.forEach(d => d.classList.remove('active'));
    currentSlide = (index + cards.length) % cards.length;
    cards[currentSlide].classList.add('active');
    dots[currentSlide].classList.add('active');
}

function startAutoSlide() {
    autoSlide = setInterval(() => showSlide(currentSlide + 1), 4500);
}

document.getElementById('nextBtn').addEventListener('click', () => {
    clearInterval(autoSlide);
    showSlide(currentSlide + 1);
    startAutoSlide();
});

document.getElementById('prevBtn').addEventListener('click', () => {
    clearInterval(autoSlide);
    showSlide(currentSlide - 1);
    startAutoSlide();
});

dots.forEach(dot => {
    dot.addEventListener('click', () => {
        clearInterval(autoSlide);
        showSlide(parseInt(dot.dataset.index));
        startAutoSlide();
    });
});

startAutoSlide();

// ─── FAQ ACCORDION ────────────────────────────────────────────────────────────
document.querySelectorAll('.faq-question').forEach(btn => {
    btn.addEventListener('click', () => {
        const item = btn.parentElement;
        const isOpen = item.classList.contains('open');

        document.querySelectorAll('.faq-item').forEach(i => i.classList.remove('open'));

        if (!isOpen) item.classList.add('open');
    });
});

// ─── CALCULATOR TABS ──────────────────────────────────────────────────────────
document.querySelectorAll('.calc-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.calc-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.calc-content').forEach(c => c.classList.add('hidden'));
        tab.classList.add('active');
        const target = tab.dataset.tab === 'gst' ? 'gstCalc' : 'incomeCalc';
        document.getElementById(target).classList.remove('hidden');
    });
});

// ─── GST CALCULATOR ───────────────────────────────────────────────────────────
function calculateGST() {
    const amount = parseFloat(document.getElementById('gstAmount').value);
    const rate = parseFloat(document.getElementById('gstRate').value);
    const type = document.querySelector('input[name="gstType"]:checked').value;

    if (!amount || amount <= 0) {
        alert('Please enter a valid amount');
        return;
    }

    let base, gstAmt, total;

    if (type === 'exclusive') {
        base = amount;
        gstAmt = (base * rate) / 100;
        total = base + gstAmt;
    } else {
        total = amount;
        base = (amount * 100) / (100 + rate);
        gstAmt = total - base;
    }

    const half = gstAmt / 2;

    document.getElementById('baseAmount').textContent = '₹' + fmt(base);
    document.getElementById('cgst').textContent = '₹' + fmt(half) + ' (' + (rate / 2) + '%)';
    document.getElementById('sgst').textContent = '₹' + fmt(half) + ' (' + (rate / 2) + '%)';
    document.getElementById('totalAmount').textContent = '₹' + fmt(total);

    const result = document.getElementById('gstResult');
    result.style.display = 'block';
    result.style.animation = 'slideIn 0.3s ease';
}

// ─── INCOME TAX CALCULATOR ────────────────────────────────────────────────────
function calculateIncomeTax() {
    const income = parseFloat(document.getElementById('incomeAmount').value);
    const regime = document.getElementById('taxRegime').value;

    if (!income || income <= 0) {
        alert('Please enter a valid income');
        return;
    }

    let tax = 0, slabDesc = '';

    if (regime === 'new') {
        // New Regime FY 2025-26 (rebate up to 7L)
        if (income <= 300000) {
            tax = 0; slabDesc = 'Nil (up to ₹3L)';
        } else if (income <= 700000) {
            tax = (income - 300000) * 0.05;
            slabDesc = '5% (₹3L – ₹7L)';
            if (income <= 700000) tax = 0; // 87A rebate
        } else if (income <= 1000000) {
            tax = (700000 - 300000) * 0.05 + (income - 700000) * 0.10;
            slabDesc = '10% (₹7L – ₹10L)';
        } else if (income <= 1200000) {
            tax = 20000 + 30000 + (income - 1000000) * 0.15;
            slabDesc = '15% (₹10L – ₹12L)';
        } else if (income <= 1500000) {
            tax = 20000 + 30000 + 30000 + (income - 1200000) * 0.20;
            slabDesc = '20% (₹12L – ₹15L)';
        } else {
            tax = 20000 + 30000 + 30000 + 60000 + (income - 1500000) * 0.30;
            slabDesc = '30% (above ₹15L)';
        }
    } else {
        // Old Regime
        if (income <= 250000) {
            tax = 0; slabDesc = 'Nil (up to ₹2.5L)';
        } else if (income <= 500000) {
            tax = (income - 250000) * 0.05;
            slabDesc = '5% (₹2.5L – ₹5L)';
            if (income <= 500000) tax = 0; // 87A rebate
        } else if (income <= 1000000) {
            tax = 12500 + (income - 500000) * 0.20;
            slabDesc = '20% (₹5L – ₹10L)';
        } else {
            tax = 112500 + (income - 1000000) * 0.30;
            slabDesc = '30% (above ₹10L)';
        }
    }

    const cess = tax * 0.04;
    const total = tax + cess;

    document.getElementById('grossIncome').textContent = '₹' + fmt(income);
    document.getElementById('taxSlab').textContent = slabDesc;
    document.getElementById('taxAmount').textContent = '₹' + fmt(tax);
    document.getElementById('cessAmount').textContent = '₹' + fmt(cess);
    document.getElementById('totalTax').textContent = '₹' + fmt(total);

    const result = document.getElementById('incomeResult');
    result.style.display = 'block';
    result.style.animation = 'slideIn 0.3s ease';
}

// Number formatter
function fmt(n) {
    return Number(n.toFixed(2)).toLocaleString('en-IN');
}

// ─── INQUIRY FORM ─────────────────────────────────────────────────────────────
const inquiryForm = document.getElementById('inquiryForm');
if (inquiryForm) {
    inquiryForm.addEventListener('submit', function(e) {
        e.preventDefault();

        const name = document.getElementById('name').value.trim();
        const email = document.getElementById('email').value.trim();
        const phone = document.getElementById('phone').value.trim();
        const service = document.getElementById('service').value;
        const message = document.getElementById('message').value.trim();

        if (!name || !email || !phone || !service || !message) {
            alert('Please fill in all required fields');
            return;
        }

        const btn = this.querySelector('.submit-btn');
        btn.textContent = 'Sending...';
        btn.disabled = true;

        setTimeout(() => {
            const success = document.createElement('div');
            success.className = 'form-success-message';
            success.innerHTML = '&#10003; Thank you, ' + name + '! Your inquiry has been sent. We\'ll respond within 24 hours.';
            this.insertAdjacentElement('beforebegin', success);

            this.reset();
            btn.textContent = 'Send Inquiry 🚀';
            btn.disabled = false;

            setTimeout(() => success.remove(), 5000);
        }, 900);
    });
}

// ─── CONTACT LINK HOVER ───────────────────────────────────────────────────────
document.querySelectorAll('.contact-card a').forEach(link => {
    link.addEventListener('mouseenter', function() {
        this.style.transform = 'scale(1.04)';
        this.style.display = 'inline-block';
        this.style.transition = 'transform 0.2s ease';
    });
    link.addEventListener('mouseleave', function() {
        this.style.transform = 'scale(1)';
    });
});
