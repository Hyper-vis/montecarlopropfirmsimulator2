import { LogoutButton } from '@/components/LogoutButton';

export default function NinjaTraderSimulator() {
    return (
        <div style={{ width: '100%', height: '100vh', display: 'flex', flexDirection: 'column', background: '#0B0B0F' }}>
            <div style={{
                background: 'rgba(100, 200, 255, 0.15)',
                border: '1px solid rgba(100, 200, 255, 0.45)',
                borderRadius: '8px',
                padding: '0.8rem 1rem',
                margin: '1rem',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                fontSize: '0.9rem'
            }}>
                <span style={{ color: '#64c8ff', fontWeight: 600 }}>⚙️ NinjaTrader Prop Account Mode</span>
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
                    <a href="/simulator/tradingview" style={{
                        fontSize: '0.78rem',
                        padding: '0.3rem 0.7rem',
                        borderRadius: '6px',
                        border: '1px solid rgba(255, 255, 255, 0.08)',
                        background: 'transparent',
                        color: '#666',
                        textDecoration: 'none',
                        cursor: 'pointer',
                        transition: 'all 0.15s'
                    }}>Switch to TradingView</a>
                    <LogoutButton />
                </div>
            </div>
            <iframe
                src="/simulator-ui.html?v=20260324f&uploadMode=ninjatrader"
                style={{ flex: 1, border: 'none', borderRadius: '8px', margin: '0 1rem 1rem 1rem' }}
            />
        </div>
    );
}
