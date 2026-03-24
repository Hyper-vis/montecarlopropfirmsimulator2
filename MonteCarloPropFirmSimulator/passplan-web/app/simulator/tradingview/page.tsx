import { LogoutButton } from '@/components/LogoutButton';

export default function TradingViewSimulator() {
    return (
        <div style={{ width: '100%', height: '100vh', display: 'flex', flexDirection: 'column', background: '#0B0B0F' }}>
            <div style={{
                background: 'rgba(138, 43, 226, 0.15)',
                border: '1px solid rgba(138, 43, 226, 0.45)',
                borderRadius: '8px',
                padding: '0.8rem 1rem',
                margin: '1rem',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                fontSize: '0.9rem'
            }}>
                <span style={{ color: '#c991ff', fontWeight: 600 }}>📊 TradingView Prop Account Mode</span>
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
                    <a href="/simulator/mt5" style={{
                        fontSize: '0.78rem',
                        padding: '0.3rem 0.7rem',
                        borderRadius: '6px',
                        border: '1px solid rgba(255, 255, 255, 0.08)',
                        background: 'transparent',
                        color: '#666',
                        textDecoration: 'none',
                        cursor: 'pointer',
                        transition: 'all 0.15s'
                    }}>Switch to MT5</a>
                    <a href="/simulator/ninjatrader" style={{
                        fontSize: '0.78rem',
                        padding: '0.3rem 0.7rem',
                        borderRadius: '6px',
                        border: '1px solid rgba(255, 255, 255, 0.08)',
                        background: 'transparent',
                        color: '#666',
                        textDecoration: 'none',
                        cursor: 'pointer',
                        transition: 'all 0.15s'
                    }}>Switch to NinjaTrader</a>
                    <LogoutButton />
                </div>
            </div>
            <iframe
                src="/simulator-ui.html?v=20260324f&uploadMode=tradingview"
                style={{ flex: 1, border: 'none', borderRadius: '8px', margin: '0 1rem 1rem 1rem' }}
            />
        </div>
    );
}
