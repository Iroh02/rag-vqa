// ===== Chart.js Defaults =====
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";

// ===== Ablation Chart =====
const ablationCtx = document.getElementById('ablationChart').getContext('2d');
new Chart(ablationCtx, {
    type: 'bar',
    data: {
        labels: [
            'Baseline',
            'Text-only\n\u03B1=0.0',
            'Balanced\n\u03B1=0.5',
            'Balanced\n+gate \u03C4=0.5',
            'Image-only\n\u03B1=1.0',
            'Image+gate\n\u03C4=0.75'
        ],
        datasets: [{
            label: 'VQA Accuracy (%)',
            data: [42.33, 41.33, 46.33, 47.33, 51.33, 49.33],
            backgroundColor: [
                'rgba(56, 189, 248, 0.6)',
                'rgba(248, 113, 113, 0.6)',
                'rgba(129, 140, 248, 0.6)',
                'rgba(129, 140, 248, 0.6)',
                'rgba(52, 211, 153, 0.7)',
                'rgba(52, 211, 153, 0.5)',
            ],
            borderColor: [
                'rgba(56, 189, 248, 1)',
                'rgba(248, 113, 113, 1)',
                'rgba(129, 140, 248, 1)',
                'rgba(129, 140, 248, 1)',
                'rgba(52, 211, 153, 1)',
                'rgba(52, 211, 153, 0.8)',
            ],
            borderWidth: 2,
            borderRadius: 6,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: { display: false },
            title: {
                display: true,
                text: 'VQA Accuracy by Configuration',
                font: { size: 14, weight: '600' },
                padding: { bottom: 16 }
            }
        },
        scales: {
            y: {
                beginAtZero: false,
                min: 35,
                max: 55,
                ticks: {
                    callback: v => v + '%',
                    stepSize: 5
                },
                grid: { color: 'rgba(255,255,255,0.04)' }
            },
            x: {
                ticks: { font: { size: 10 } },
                grid: { display: false }
            }
        }
    }
});

// ===== Error Analysis Chart =====
const errorCtx = document.getElementById('errorChart').getContext('2d');
new Chart(errorCtx, {
    type: 'doughnut',
    data: {
        labels: [
            'Model Ignored Context (63%)',
            'Wrong Images Retrieved (20%)',
            'Misleading Answers (17%)'
        ],
        datasets: [{
            data: [29, 9, 8],
            backgroundColor: [
                'rgba(248, 113, 113, 0.7)',
                'rgba(251, 191, 36, 0.7)',
                'rgba(59, 130, 246, 0.7)',
            ],
            borderColor: [
                'rgba(248, 113, 113, 1)',
                'rgba(251, 191, 36, 1)',
                'rgba(59, 130, 246, 1)',
            ],
            borderWidth: 2,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            title: {
                display: true,
                text: 'Error Breakdown (46 incorrect predictions)',
                font: { size: 14, weight: '600' },
                padding: { bottom: 16 }
            },
            legend: {
                position: 'bottom',
                labels: { font: { size: 11 }, padding: 12 }
            }
        }
    }
});

// ===== Training Loss Chart =====
const lossCtx = document.getElementById('lossChart').getContext('2d');
new Chart(lossCtx, {
    type: 'line',
    data: {
        labels: ['Epoch 1', 'Epoch 2', 'Epoch 3'],
        datasets: [{
            label: 'Avg Training Loss',
            data: [3.99, 2.94, 2.70],
            borderColor: 'rgba(56, 189, 248, 1)',
            backgroundColor: 'rgba(56, 189, 248, 0.1)',
            fill: true,
            tension: 0.3,
            pointRadius: 5,
            pointBackgroundColor: 'rgba(56, 189, 248, 1)',
            borderWidth: 2,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: { display: false },
            title: {
                display: true,
                text: 'Training Loss',
                font: { size: 12, weight: '600' }
            }
        },
        scales: {
            y: {
                min: 2.0,
                max: 4.5,
                grid: { color: 'rgba(255,255,255,0.04)' }
            },
            x: {
                grid: { display: false }
            }
        }
    }
});

// ===== Inference Traces =====
// Real data: baseline predictions vs RAG predictions vs ground truth
const traces = [
    {
        question: "Is the skier going downhill?",
        qid: 2532000,
        gt: "yes",
        baseline: "no",
        rag: "yes",
        mode: "correct",
        modeLabel: "RAG helped",
        retrieved: [
            { question: "Are they doing the same sport?", answer: "yes", score: 0.82 },
            { question: "Has this area been windy lately?", answer: "yes", score: 0.82 },
            { question: "Is this the morning?", answer: "yes", score: 0.82 }
        ]
    },
    {
        question: "Is there snow on the ground?",
        qid: 2529010,
        gt: "no",
        baseline: "no",
        rag: "no",
        mode: "correct",
        modeLabel: "Both correct",
        retrieved: [
            { question: "How tall is the house?", answer: "2 stories", score: 1.00 },
            { question: "Is it 4am or 4pm?", answer: "pm", score: 1.00 },
            { question: "What kind of buildings?", answer: "houses", score: 1.00 }
        ]
    },
    {
        question: "What color is the sky?",
        qid: 2532004,
        gt: "blue",
        baseline: "blue",
        rag: "blue",
        mode: "correct",
        modeLabel: "Both correct",
        retrieved: [
            { question: "Are they doing the same sport?", answer: "yes", score: 0.82 },
            { question: "Has this area been windy lately?", answer: "yes", score: 0.82 },
            { question: "Is this the morning?", answer: "yes", score: 0.82 }
        ]
    },
    {
        question: "How cold is it?",
        qid: 2532002,
        gt: "very cold",
        baseline: "sky is blue",
        rag: "very cold",
        mode: "correct",
        modeLabel: "RAG helped",
        retrieved: [
            { question: "Are they doing the same sport?", answer: "yes", score: 0.82 },
            { question: "Has this area been windy lately?", answer: "yes", score: 0.82 },
            { question: "Is this the morning?", answer: "yes", score: 0.82 }
        ]
    },
    {
        question: "Is it a sunny or cloudy day?",
        qid: 2532003,
        gt: "sunny",
        baseline: "sunny day",
        rag: "yes",
        mode: "ignored",
        modeLabel: "Model ignored context",
        retrieved: [
            { question: "Are they doing the same sport?", answer: "yes", score: 0.82 },
            { question: "Has this area been windy lately?", answer: "yes", score: 0.82 },
            { question: "Is this the morning?", answer: "yes", score: 0.82 }
        ]
    },
    {
        question: "What filter is being used in this image?",
        qid: 395749000,
        gt: "black and white",
        baseline: "black and white",
        rag: "black and white",
        mode: "correct",
        modeLabel: "Both correct",
        retrieved: [
            { question: "What is the girl sitting on?", answer: "grass", score: 0.79 },
            { question: "What are the girls doing?", answer: "sitting", score: 0.79 },
            { question: "How many people are watching the water?", answer: "2", score: 0.79 }
        ]
    },
    {
        question: "How many people can be seen?",
        qid: 395749002,
        gt: "3",
        baseline: "Three",
        rag: "3",
        mode: "correct",
        modeLabel: "Both correct",
        retrieved: [
            { question: "What is the girl sitting on?", answer: "grass", score: 0.79 },
            { question: "What are the girls doing?", answer: "sitting", score: 0.79 },
            { question: "How many people are watching the water?", answer: "2", score: 0.79 }
        ]
    },
    {
        question: "What ocean might this be?",
        qid: 395749003,
        gt: "atlantic",
        baseline: "Atlantic Ocean",
        rag: "ocean",
        mode: "misleading",
        modeLabel: "Misleading context",
        retrieved: [
            { question: "What is the girl sitting on?", answer: "grass", score: 0.79 },
            { question: "What are the girls doing?", answer: "sitting", score: 0.79 },
            { question: "How many people are watching the water?", answer: "2", score: 0.79 }
        ]
    }
];

const tracesContainer = document.getElementById('traces-container');
traces.forEach(t => {
    const details = document.createElement('details');
    details.className = 'trace-card';

    const modeClass = t.mode;
    const baselineMatch = t.baseline.toLowerCase().replace(/[^\w\s]/g, '').trim() ===
                          t.gt.toLowerCase().replace(/[^\w\s]/g, '').trim();
    const ragMatch = t.rag.toLowerCase().replace(/[^\w\s]/g, '').trim() ===
                     t.gt.toLowerCase().replace(/[^\w\s]/g, '').trim();

    let hintsHTML = '';
    t.retrieved.forEach(r => {
        hintsHTML += `<div class="hint-item">Q: ${r.question} &rarr; <strong>${r.answer}</strong> <span class="hint-score">${r.score.toFixed(2)}</span></div>`;
    });

    details.innerHTML = `
        <summary>
            <span class="mode-pill ${modeClass}">${t.modeLabel}</span>
            <span>${t.question}</span>
        </summary>
        <div class="trace-body">
            <div class="trace-row">
                <span class="trace-label">Ground Truth</span>
                <span class="trace-val">${t.gt}</span>
            </div>
            <div class="trace-row">
                <span class="trace-label">Baseline</span>
                <span class="trace-val ${baselineMatch ? 'correct' : 'wrong'}">${t.baseline} ${baselineMatch ? '\u2713' : '\u2717'}</span>
            </div>
            <div class="trace-row">
                <span class="trace-label">RAG (\u03B1=1.0)</span>
                <span class="trace-val ${ragMatch ? 'correct' : 'wrong'}">${t.rag} ${ragMatch ? '\u2713' : '\u2717'}</span>
            </div>
            <div class="trace-hints">
                <div style="font-weight:600; color:#94a3b8; margin-bottom:0.3rem; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.03em;">Retrieved Hints (top-3)</div>
                ${hintsHTML}
            </div>
        </div>
    `;
    tracesContainer.appendChild(details);
});

// ===== Demo Gallery =====
const demoExamples = [
    {
        title: "RAG helps — airline recognition",
        category: "helps",
        queryImage: "assets/demo/q1.jpg",
        question: "What airline does the plane fly for?",
        baseline: "United Airlines",
        rag: "American Airlines",
        gt: "American Airlines",
        retrieved: [
            { image: "assets/demo/q1_r1.jpg", question: "How many engines does the plane have?", answer: "4", score: 0.85 },
            { image: "assets/demo/q1_r2.jpg", question: "How many planes?", answer: "1", score: 0.85 },
            { image: "assets/demo/q1_r3.jpg", question: "Is the plane at an airport?", answer: "yes", score: 0.85 }
        ]
    },
    {
        title: "RAG helps — food identification",
        category: "helps",
        queryImage: "assets/demo/q2.jpg",
        question: "How are the eggs cooked?",
        baseline: "not",
        rag: "scrambled",
        gt: "scrambled",
        retrieved: [
            { image: "assets/demo/q2_r1.jpg", question: "What is this?", answer: "food", score: 0.82 },
            { image: "assets/demo/q2_r2.jpg", question: "Is this a meal for a child?", answer: "no", score: 0.82 },
            { image: "assets/demo/q2_r3.jpg", question: "Is this a sandwich with a lot of bread?", answer: "no", score: 0.82 }
        ]
    },
    {
        title: "RAG helps — time of day",
        category: "helps",
        queryImage: "assets/demo/q3.jpg",
        question: "Is it morning or evening?",
        baseline: "morning",
        rag: "evening",
        gt: "evening",
        retrieved: [
            { image: "assets/demo/q3_r1.jpg", question: "Are street lights on?", answer: "yes", score: 0.80 },
            { image: "assets/demo/q3_r2.jpg", question: "Is this a night or daytime shot?", answer: "night", score: 0.80 },
            { image: "assets/demo/q3_r3.jpg", question: "What are the long lines of light?", answer: "car lights", score: 0.80 }
        ]
    },
    {
        title: "RAG helps — scene understanding",
        category: "helps",
        queryImage: "assets/demo/q4.jpg",
        question: "Is it sunny out?",
        baseline: "yes",
        rag: "no",
        gt: "no",
        retrieved: [
            { image: "assets/demo/q4_r1.jpg", question: "Are these elephants inside or outside?", answer: "no", score: 0.78 },
            { image: "assets/demo/q4_r2.jpg", question: "Does the smaller elephant have tusks?", answer: "yes", score: 0.78 },
            { image: "assets/demo/q4_r3.jpg", question: "Are all of the elephants the same size?", answer: "no", score: 0.78 }
        ]
    },
    {
        title: "RAG ignored — either/or question",
        category: "ignored",
        queryImage: "assets/demo/q5.jpg",
        question: "Is it a sunny or cloudy day?",
        baseline: "sunny day",
        rag: "yes",
        gt: "sunny",
        retrieved: [
            { image: "assets/demo/q5_r1.jpg", question: "Are they doing the same sport?", answer: "yes", score: 0.82 },
            { image: "assets/demo/q5_r2.jpg", question: "Has this area been windy lately?", answer: "yes", score: 0.82 },
            { image: "assets/demo/q5_r3.jpg", question: "Is this the morning?", answer: "yes", score: 0.82 }
        ]
    },
    {
        title: "RAG helps — reading details",
        category: "helps",
        queryImage: "assets/demo/q6.jpg",
        question: "How much time is left?",
        baseline: "None",
        rag: "15 minutes",
        gt: "15 minutes",
        retrieved: [
            { image: "assets/demo/q6_r1.jpg", question: "Could you see someone clearly in the reflection?", answer: "no", score: 0.77 },
            { image: "assets/demo/q6_r2.jpg", question: "Is the water turned on or off?", answer: "off", score: 0.77 },
            { image: "assets/demo/q6_r3.jpg", question: "Where is the electrical outlet?", answer: "wall", score: 0.77 }
        ]
    }
];

const demoContainer = document.getElementById('demo-container');
demoExamples.forEach(d => {
    const details = document.createElement('details');
    details.className = 'demo-card';

    const norm = s => s.toLowerCase().replace(/[^\w\s]/g, '').replace(/\b(a|an|the)\b/g, '').trim();
    const blMatch = norm(d.baseline) === norm(d.gt);
    const ragMatch = norm(d.rag) === norm(d.gt);

    let retrievedHTML = '';
    d.retrieved.forEach(r => {
        retrievedHTML += `
            <div class="demo-retrieved-card">
                <img class="demo-retrieved-img" src="${r.image}" alt="Retrieved" loading="lazy">
                <div class="demo-retrieved-caption">
                    Q: ${r.question}<br>
                    A: <strong>${r.answer}</strong>
                    <span class="demo-r-score">${r.score.toFixed(2)}</span>
                </div>
            </div>`;
    });

    details.innerHTML = `
        <summary>
            <span class="demo-category ${d.category}">${d.category}</span>
            <span>${d.title}</span>
        </summary>
        <div class="demo-body">
            <div class="demo-layout">
                <div>
                    <img class="demo-query-img" src="${d.queryImage}" alt="Query image" loading="lazy">
                </div>
                <div class="demo-info">
                    <div class="demo-question">${d.question}</div>
                    <div class="demo-answers">
                        <div class="demo-answer-row">
                            <span class="demo-answer-label">Ground truth</span>
                            <span class="demo-pill gt">${d.gt}</span>
                        </div>
                        <div class="demo-answer-row">
                            <span class="demo-answer-label">Baseline</span>
                            <span class="demo-pill baseline ${blMatch ? 'correct-match' : 'wrong-match'}">${d.baseline}</span>
                            <span class="demo-check">${blMatch ? '\u2713' : '\u2717'}</span>
                        </div>
                        <div class="demo-answer-row">
                            <span class="demo-answer-label">RAG</span>
                            <span class="demo-pill ${ragMatch ? 'rag-correct' : 'rag-wrong'}">${d.rag}</span>
                            <span class="demo-check">${ragMatch ? '\u2713' : '\u2717'}</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="demo-retrieved-section">
                <div class="demo-retrieved-title">Retrieved from KB (top-3, \u03B1=1.0 image similarity)</div>
                <div class="demo-retrieved-grid">
                    ${retrievedHTML}
                </div>
            </div>
        </div>
    `;
    demoContainer.appendChild(details);
});

// ===== Example Predictions =====
const examples = [
    {
        question: "Is there snow on the ground?",
        answer: "no",
        rag_used: true,
        retrieved: [
            { question: "How tall is the house?", answer: "2 stories", score: 1.0 },
            { question: "Is it 4am or 4pm?", answer: "pm", score: 1.0 },
            { question: "What kind of buildings?", answer: "houses", score: 1.0 }
        ]
    },
    {
        question: "Is the skier going downhill?",
        answer: "yes",
        rag_used: true,
        retrieved: [
            { question: "Are they doing the same sport?", answer: "yes", score: 0.82 },
            { question: "Has this area been windy?", answer: "yes", score: 0.82 },
            { question: "Is this the morning?", answer: "yes", score: 0.82 }
        ]
    },
    {
        question: "What color is the bus?",
        answer: "red",
        rag_used: true,
        retrieved: [
            { question: "What type of vehicle?", answer: "bus", score: 0.91 },
            { question: "What color is the bus?", answer: "red", score: 0.88 },
            { question: "Is the bus moving?", answer: "yes", score: 0.85 }
        ]
    },
    {
        question: "How many people are in the photo?",
        answer: "2",
        rag_used: true,
        retrieved: [
            { question: "How many people?", answer: "3", score: 0.79 },
            { question: "What are they doing?", answer: "eating", score: 0.76 },
            { question: "Is this outdoors?", answer: "no", score: 0.74 }
        ]
    },
    {
        question: "What animal is shown?",
        answer: "cat",
        rag_used: true,
        retrieved: [
            { question: "What type of animal?", answer: "cat", score: 0.95 },
            { question: "What color is the cat?", answer: "orange", score: 0.93 },
            { question: "Is the cat sleeping?", answer: "yes", score: 0.90 }
        ]
    },
    {
        question: "What sport is being played?",
        answer: "tennis",
        rag_used: true,
        retrieved: [
            { question: "What sport?", answer: "tennis", score: 0.88 },
            { question: "Is this indoors?", answer: "no", score: 0.85 },
            { question: "What is she holding?", answer: "racket", score: 0.83 }
        ]
    }
];

const container = document.getElementById('examples-container');
examples.forEach(ex => {
    const card = document.createElement('div');
    card.className = 'example-card';

    let retrievedHTML = '';
    ex.retrieved.forEach(r => {
        retrievedHTML += `
            <div class="retrieved-item">
                Q: ${r.question} &rarr; <strong>${r.answer}</strong>
                <span class="score">${r.score.toFixed(2)}</span>
            </div>`;
    });

    card.innerHTML = `
        <div class="question">${ex.question}</div>
        <div class="answer">${ex.answer}</div>
        <div class="rag-label">Retrieved Context (${ex.retrieved.length} examples)</div>
        ${retrievedHTML}
    `;
    container.appendChild(card);
});

// ===== Copy Command Button =====
function copyCmd(btn) {
    const code = btn.previousElementSibling.textContent;
    navigator.clipboard.writeText(code).then(() => {
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
    });
}

// ===== Sticky Nav Active State =====
const sections = document.querySelectorAll('.section');
const navLinks = document.querySelectorAll('.nav-inner a');

const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            navLinks.forEach(link => {
                link.classList.toggle('active',
                    link.getAttribute('href') === '#' + entry.target.id);
            });
        }
    });
}, { rootMargin: '-50% 0px -50% 0px' });

sections.forEach(section => observer.observe(section));
