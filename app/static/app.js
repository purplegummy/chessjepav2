document.addEventListener('DOMContentLoaded', () => {

    // ── State ────────────────────────────────────────────────────────────────
    const game      = new Chess();
    let board       = null;
    let playerColor = 'white';
    let selectedSq  = null;
    let hintsOn     = true;
    let aiPending   = false;
    let lastAiMove  = null;

    // ── DOM refs ─────────────────────────────────────────────────────────────
    const statusCard    = document.getElementById('statusCard');
    const statusIcon    = document.getElementById('statusIcon');
    const statusTitle   = document.getElementById('statusTitle');
    const statusSub     = document.getElementById('statusSub');
    const statusBadge   = document.getElementById('statusBadge');
    const thinkingDot   = document.getElementById('thinkingDot');
    const bestMoveSan   = document.getElementById('bestMoveSan');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidencePct = document.getElementById('confidencePct');
    const topMovesList  = document.getElementById('topMovesList');
    const moveHistory   = document.getElementById('moveHistory');
    const fenInput      = document.getElementById('fenInput');
    const boardPanel    = document.querySelector('.board-panel');

    // ── Chessboard.js config ─────────────────────────────────────────────────
    function buildBoardConfig() {
        return {
            pieceTheme: function(piece) {
                const glyphs = {
                    wK:'♔', wQ:'♕', wR:'♖', wB:'♗', wN:'♘', wP:'♙',
                    bK:'♚', bQ:'♛', bR:'♜', bB:'♝', bN:'♞', bP:'♟'
                };
                const color  = piece[0] === 'w' ? '#fff' : '#1a1a1a';
                const shadow = piece[0] === 'w'
                    ? 'drop-shadow(0 1px 3px rgba(0,0,0,0.8))'
                    : 'drop-shadow(0 1px 2px rgba(255,255,255,0.3))';
                return `data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 45 45'><text y='36' x='2' font-size='38' fill='${encodeURIComponent(color)}' style='filter:${encodeURIComponent(shadow)}'>${encodeURIComponent(glyphs[piece])}</text></svg>`;
            },
            position:    'start',
            draggable:   true,
            orientation: playerColor,

            onDragStart(source, piece) {
                if (game.game_over() || aiPending) return false;
                if (game.turn() === 'w' && piece.charAt(0) === 'b') return false;
                if (game.turn() === 'b' && piece.charAt(0) === 'w') return false;
                if (game.turn() !== playerColor[0]) return false;
                if (hintsOn) highlightLegal(source);
                return true;
            },

            onDrop(source, target) {
                clearHighlights();
                const move = tryMove(source, target);
                if (!move) return 'snapback';
            },

            onSnapEnd() {
                board.position(game.fen());
            },
        };
    }

    board = Chessboard('board', buildBoardConfig());

    // ── Move logic ───────────────────────────────────────────────────────────
    function tryMove(from, to) {
        const move = game.move({ from, to, promotion: 'q' });
        if (!move) return null;
        lastAiMove = null;
        updateAfterMove();
        if (!game.game_over()) scheduleAiMove();
        return move;
    }

    function scheduleAiMove() {
        aiPending = true;
        boardPanel.classList.add('disabled');
        thinkingDot.classList.add('active');
        updateStatus();
        setTimeout(makeAiMove, 300);
    }

    async function makeAiMove() {
        try {
            const resp = await fetch('/api/best_move', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ fen: game.fen(), top_n: 5 }),
            });

            if (!resp.ok) throw new Error(`Server error ${resp.status}`);
            const data = await resp.json();

            const uci   = data.move;
            const from  = uci.slice(0, 2);
            const to    = uci.slice(2, 4);
            const promo = uci.length === 5 ? uci[4] : 'q';

            game.move({ from, to, promotion: promo });
            lastAiMove = { from, to };
            board.position(game.fen());
            updateAnalysisPanel(data);
        } catch (err) {
            console.error('AI move error:', err);
            clearAnalysisPanel();
        } finally {
            aiPending = false;
            boardPanel.classList.remove('disabled');
            thinkingDot.classList.remove('active');
            updateAfterMove();
        }
    }

    // ── Highlight helpers ────────────────────────────────────────────────────
    function highlightLegal(source) {
        clearHighlights();
        game.moves({ square: source, verbose: true }).forEach(m => {
            document.querySelector(`[data-square="${m.to}"]`)?.classList.add('sq-legal');
        });
        document.querySelector(`[data-square="${source}"]`)?.classList.add('sq-selected');
    }

    function clearHighlights() {
        document.querySelectorAll('.sq-legal, .sq-selected, .sq-ai').forEach(el => {
            el.classList.remove('sq-legal', 'sq-selected', 'sq-ai');
        });
    }

    function highlightAiMove() {
        clearHighlights();
        if (!lastAiMove) return;
        document.querySelector(`[data-square="${lastAiMove.from}"]`)?.classList.add('sq-ai');
        document.querySelector(`[data-square="${lastAiMove.to}"]`)?.classList.add('sq-ai');
    }

    // ── UI updates ───────────────────────────────────────────────────────────
    function updateAfterMove() {
        updateStatus();
        updateMoveHistory();
        fenInput.value = game.fen();
        highlightAiMove();
    }

    function updateStatus() {
        statusCard.className  = 'status-card';
        statusBadge.className = 'status-badge';

        if (game.in_checkmate()) {
            const winner      = game.turn() === 'w' ? 'Black' : 'White';
            const isPlayerWin = winner.toLowerCase() === playerColor;
            statusIcon.textContent  = isPlayerWin ? '🏆' : '🤖';
            statusTitle.textContent = isPlayerWin ? 'You Win!' : 'JEPA Wins!';
            statusSub.textContent   = `Checkmate — ${winner} wins`;
            statusBadge.textContent = 'Game Over';
            statusCard.classList.add('game-over');
            statusBadge.classList.add('badge-gameover');
            return;
        }
        if (game.in_stalemate() || game.in_draw()) {
            statusIcon.textContent  = '🤝';
            statusTitle.textContent = 'Draw';
            statusSub.textContent   = game.in_stalemate() ? 'Stalemate' : 'Draw';
            statusBadge.textContent = 'Draw';
            statusCard.classList.add('game-over');
            statusBadge.classList.add('badge-draw');
            return;
        }
        if (game.in_check()) {
            statusIcon.textContent  = '⚠️';
            statusTitle.textContent = 'Check!';
            statusSub.textContent   = `${game.turn() === 'w' ? 'White' : 'Black'} is in check`;
            statusBadge.textContent = 'Check';
            statusCard.classList.add('in-check');
            statusBadge.classList.add('badge-check');
            return;
        }
        if (aiPending) {
            statusIcon.textContent  = '🤖';
            statusTitle.textContent = 'JEPA Thinking…';
            statusSub.textContent   = 'Running policy head';
            statusBadge.textContent = 'AI';
            statusCard.classList.add('ai-turn');
            statusBadge.classList.add('badge-ai');
        } else {
            statusIcon.textContent  = '♟';
            statusTitle.textContent = 'Your Turn';
            statusSub.textContent   = `${game.turn() === 'w' ? 'White' : 'Black'} to move`;
            statusBadge.textContent = 'Playing';
            statusCard.classList.add('your-turn');
            statusBadge.classList.add('badge-playing');
        }
    }

    function updateAnalysisPanel(data) {
        bestMoveSan.textContent = data.san || data.move || '—';

        const pct = ((data.confidence || 0) * 100).toFixed(1);
        confidenceBar.style.width = `${pct}%`;
        confidencePct.textContent = `${pct}%`;

        topMovesList.innerHTML = '';
        (data.top_moves || []).forEach((m, i) => {
            const pctStr = ((m.prob || 0) * 100).toFixed(1);
            const div = document.createElement('div');
            div.className = `move-row${i === 0 ? ' best' : ''}`;
            div.innerHTML = `
                <span class="move-rank">#${i + 1}</span>
                <span class="move-san">${m.san}</span>
                <div class="move-bar-wrap">
                    <div class="move-bar-fill" style="width:${pctStr}%"></div>
                </div>
                <span class="move-prob">${pctStr}%</span>
            `;
            topMovesList.appendChild(div);
        });
    }

    function clearAnalysisPanel() {
        bestMoveSan.textContent   = '—';
        confidenceBar.style.width = '0%';
        confidencePct.textContent = '—';
        topMovesList.innerHTML    = '';
    }

    function updateMoveHistory() {
        const history = game.history({ verbose: true });
        if (!history.length) {
            moveHistory.innerHTML = '<span class="history-empty">No moves yet</span>';
            return;
        }
        let html = '';
        for (let i = 0; i < history.length; i += 2) {
            const moveNum = Math.floor(i / 2) + 1;
            const wSan    = history[i].san;
            const bSan    = history[i + 1] ? history[i + 1].san : null;
            const wLast   = i === history.length - 1;
            const bLast   = i + 1 === history.length - 1;
            html += `<span class="history-pair">
                <span class="history-num">${moveNum}.</span>
                <span class="history-san${wLast ? ' last' : ''}">${wSan}</span>
                ${bSan ? `<span class="history-san${bLast ? ' last' : ''}">${bSan}</span>` : ''}
            </span>`;
        }
        moveHistory.innerHTML = html;
        moveHistory.scrollTop = moveHistory.scrollHeight;
    }

    // ── New Game ─────────────────────────────────────────────────────────────
    function newGame() {
        game.reset();
        board.position('start');
        board.orientation(playerColor);
        lastAiMove = null;
        aiPending  = false;
        boardPanel.classList.remove('disabled');
        thinkingDot.classList.remove('active');
        clearHighlights();
        clearAnalysisPanel();
        updateAfterMove();
        if (playerColor === 'black') scheduleAiMove();
    }

    // ── Buttons ──────────────────────────────────────────────────────────────
    document.getElementById('newGameBtn').addEventListener('click', newGame);

    document.getElementById('takeBackBtn').addEventListener('click', () => {
        if (aiPending) return;
        game.undo(); game.undo();
        lastAiMove = null;
        board.position(game.fen());
        clearHighlights();
        clearAnalysisPanel();
        updateAfterMove();
    });

    document.getElementById('flipBoardBtn').addEventListener('click', () => board.flip());

    document.querySelectorAll('#colorToggle .toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#colorToggle .toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            playerColor = btn.dataset.color;
            newGame();
        });
    });

    document.getElementById('hintsToggle').addEventListener('change', e => {
        hintsOn = e.target.checked;
        if (!hintsOn) clearHighlights();
    });

    // ── Init ─────────────────────────────────────────────────────────────────
    updateAfterMove();
});
