document.addEventListener('DOMContentLoaded', () => {
    const navbar = document.querySelector('.app-navbar');
    if (navbar) {
        const toggleNavState = () => {
            navbar.classList.toggle('is-scrolled', window.scrollY > 10);
        };
        toggleNavState();
        window.addEventListener('scroll', toggleNavState);
    }

    const animatedBlocks = document.querySelectorAll('[data-animate]');
    if (animatedBlocks.length) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('is-visible');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.18 });

        animatedBlocks.forEach(block => observer.observe(block));
    }

    if (window.bootstrap) {
        document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
            new bootstrap.Tooltip(el);
        });
    }

    document.querySelectorAll('[data-password-toggle]').forEach(toggleBtn => {
        const input = document.getElementById(toggleBtn.dataset.passwordToggle);
        if (!input) return;

        toggleBtn.addEventListener('click', () => {
            const isHidden = input.type === 'password';
            input.type = isHidden ? 'text' : 'password';
            toggleBtn.setAttribute('aria-pressed', isHidden ? 'true' : 'false');
            const icon = toggleBtn.querySelector('i');
            if (icon) {
                icon.classList.toggle('fa-eye', !isHidden);
                icon.classList.toggle('fa-eye-slash', isHidden);
            }
        });
    });

    const evaluateStrength = (value) => {
        let score = 0;
        if (value.length >= 8) score += 1;
        if (/[A-Z]/.test(value) && /[a-z]/.test(value)) score += 1;
        if (/\d/.test(value) || /[^A-Za-z0-9]/.test(value)) score += 1;

        if (!value) return '';
        if (score === 1) return 'weak';
        if (score === 2) return 'medium';
        if (score >= 3) return 'strong';
        return '';
    };

    document.querySelectorAll('[data-strength-target]').forEach(input => {
        const meter = document.getElementById(input.dataset.strengthTarget);
        if (!meter) return;

        const updateMeter = () => {
            meter.dataset.level = evaluateStrength(input.value);
        };

        input.addEventListener('input', updateMeter);
        updateMeter();
    });
});

