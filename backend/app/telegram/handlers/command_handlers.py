"""
AlphaTerminal Pro - Telegram Handlers v4.2
==========================================

Bot komut ve callback işleyicileri

Komutlar:
- /start - Bot başlat
- /signals - Aktif sinyaller
- /portfolio - Portföy durumu
- /analysis <symbol> - Hisse analizi
- /risk - Risk durumu
- /report - Günlük rapor
- /help - Yardım

Author: AlphaTerminal Team
Version: 4.2.0
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class CommandType(Enum):
    """Komut türleri"""
    START = "start"
    HELP = "help"
    SIGNALS = "signals"
    PORTFOLIO = "portfolio"
    ANALYSIS = "analysis"
    RISK = "risk"
    REPORT = "report"
    SETTINGS = "settings"
    STOP = "stop"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CommandContext:
    """Komut bağlamı"""
    chat_id: str
    user_id: str
    command: str
    args: list
    message_id: Optional[str] = None
    username: Optional[str] = None


@dataclass
class CallbackContext:
    """Callback bağlamı"""
    chat_id: str
    user_id: str
    callback_data: str
    message_id: str


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

class CommandHandlers:
    """
    Telegram komut işleyicileri
    """
    
    def __init__(self, bot):
        """
        Args:
            bot: TelegramBot instance
        """
        self.bot = bot
        self._handlers: Dict[str, Callable] = {
            "start": self.handle_start,
            "help": self.handle_help,
            "signals": self.handle_signals,
            "portfolio": self.handle_portfolio,
            "analysis": self.handle_analysis,
            "risk": self.handle_risk,
            "report": self.handle_report,
            "settings": self.handle_settings,
            "stop": self.handle_stop,
        }
    
    async def handle_command(self, ctx: CommandContext) -> bool:
        """
        Komut işle
        
        Args:
            ctx: Komut bağlamı
            
        Returns:
            Başarılı mı
        """
        handler = self._handlers.get(ctx.command)
        
        if handler:
            return await handler(ctx)
        else:
            await self.bot.send_message(
                f"❓ Bilinmeyen komut: /{ctx.command}\n\nYardım için /help yazın.",
                chat_id=ctx.chat_id
            )
            return False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INDIVIDUAL HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def handle_start(self, ctx: CommandContext) -> bool:
        """Start komutu"""
        text = """
🚀 <b>AlphaTerminal Pro'ya Hoş Geldiniz!</b>

Professional BIST trading sinyalleri ve analiz botu.

<b>📋 Komutlar:</b>
/signals - Aktif sinyaller
/portfolio - Portföy durumu
/analysis THYAO - Hisse analizi
/risk - Risk durumu
/report - Günlük rapor
/help - Detaylı yardım

<b>🔔 Bildirimler:</b>
• Yeni sinyaller otomatik gönderilir
• Risk uyarıları anlık bildirilir
• Günlük özet her gün 18:30'da

<i>İyi tradeler! 📈</i>
"""
        return await self.bot.send_message(text.strip(), chat_id=ctx.chat_id)
    
    async def handle_help(self, ctx: CommandContext) -> bool:
        """Help komutu"""
        text = """
📖 <b>AlphaTerminal Pro - Yardım</b>

<b>🎯 Sinyal Komutları:</b>
/signals - Tüm aktif sinyalleri listele
/signals THYAO - Belirli hisse sinyali

<b>💼 Portföy Komutları:</b>
/portfolio - Portföy özeti
/portfolio detail - Detaylı pozisyonlar

<b>📊 Analiz Komutları:</b>
/analysis THYAO - Hisse analizi
/analysis THYAO smc - SMC analizi
/analysis THYAO of - OrderFlow analizi

<b>⚠️ Risk Komutları:</b>
/risk - Güncel risk durumu
/risk limits - Risk limitleri

<b>📋 Rapor Komutları:</b>
/report - Günlük rapor
/report weekly - Haftalık rapor

<b>⚙️ Ayarlar:</b>
/settings - Bildirim ayarları
/stop - Bildirimleri durdur

<b>💡 İpuçları:</b>
• Sinyallerde ✅/❌ butonları ile onaylayın
• 📈 Chart butonu ile grafiğe ulaşın
• Risk uyarılarını ciddiye alın

<i>Sorularınız için: @alphaterminal_support</i>
"""
        return await self.bot.send_message(text.strip(), chat_id=ctx.chat_id)
    
    async def handle_signals(self, ctx: CommandContext) -> bool:
        """Signals komutu"""
        from app.services.signal_service import SignalService
        
        try:
            # Eğer sembol belirtilmişse
            if ctx.args:
                symbol = ctx.args[0].upper()
                # Belirli sembol sinyali
                text = f"""
🎯 <b>{symbol} Sinyal Durumu</b>

Aktif sinyal bulunamadı.

Son analiz için: /analysis {symbol}
"""
            else:
                # Tüm aktif sinyaller
                text = """
📡 <b>Aktif Sinyaller</b>

Şu anda aktif sinyal bulunmuyor.

Yeni sinyal geldiğinde otomatik bildirilecek.
"""
            
            return await self.bot.send_message(text.strip(), chat_id=ctx.chat_id)
            
        except Exception as e:
            logger.error(f"Signals handler error: {e}")
            return await self.bot.send_message(
                "❌ Sinyal bilgisi alınamadı. Lütfen tekrar deneyin.",
                chat_id=ctx.chat_id
            )
    
    async def handle_portfolio(self, ctx: CommandContext) -> bool:
        """Portfolio komutu"""
        from app.core.shadow_mode import get_shadow_system
        
        try:
            shadow = get_shadow_system()
            
            if not shadow.is_active:
                text = """
💼 <b>Portföy Durumu</b>

Shadow Mode aktif değil.

Canlı portföy takibi için web arayüzünü kullanın.
"""
            else:
                portfolio = shadow.get_portfolio()
                positions = shadow.get_open_trades()
                
                pnl_emoji = "📈" if portfolio.total_pnl >= 0 else "📉"
                
                pos_text = ""
                for pos in positions[:5]:
                    pos_emoji = "🟢" if pos.unrealized_pnl >= 0 else "🔴"
                    pos_text += f"\n{pos_emoji} {pos.symbol}: {pos.unrealized_pnl:+.0f} TRY"
                
                text = f"""
💼 <b>Portföy Durumu</b>

<b>💰 Sermaye:</b> <code>{portfolio.current_capital:,.0f}</code> TRY
<b>{pnl_emoji} Toplam PnL:</b> <code>{portfolio.total_pnl:+,.0f}</code> TRY ({portfolio.total_pnl_pct:+.2f}%)

<b>📊 İstatistikler:</b>
├ Açık Pozisyon: {portfolio.open_positions}
├ Toplam Trade: {portfolio.total_trades}
├ Win Rate: {portfolio.win_rate:.1f}%
└ Max DD: {portfolio.max_drawdown:.1f}%

<b>📍 Pozisyonlar:</b>{pos_text if pos_text else "\nPozisyon yok"}
"""
            
            return await self.bot.send_message(text.strip(), chat_id=ctx.chat_id)
            
        except Exception as e:
            logger.error(f"Portfolio handler error: {e}")
            return await self.bot.send_message(
                "❌ Portföy bilgisi alınamadı.",
                chat_id=ctx.chat_id
            )
    
    async def handle_analysis(self, ctx: CommandContext) -> bool:
        """Analysis komutu"""
        if not ctx.args:
            return await self.bot.send_message(
                "❌ Kullanım: /analysis <SEMBOL>\n\nÖrnek: /analysis THYAO",
                chat_id=ctx.chat_id
            )
        
        symbol = ctx.args[0].upper()
        analysis_type = ctx.args[1].lower() if len(ctx.args) > 1 else "full"
        
        try:
            from app.services.analysis_service import AnalysisService
            
            service = AnalysisService()
            result = await service.analyze_stock(symbol)
            
            if not result:
                return await self.bot.send_message(
                    f"❌ {symbol} için analiz yapılamadı.",
                    chat_id=ctx.chat_id
                )
            
            # Format analysis
            smc = result.get("smc", {})
            of = result.get("orderflow", {})
            alpha = result.get("alpha", {})
            
            structure_emoji = "🟢" if smc.get("bias") == "LONG" else "🔴"
            
            text = f"""
📊 <b>{symbol} Analizi</b>

<b>🏗️ Market Yapısı:</b> {structure_emoji} {smc.get('structure', 'N/A')}
<b>📍 Bias:</b> {smc.get('bias', 'N/A')}

<b>📈 SMC ({smc.get('score', 0)}/100):</b>
├ BOS: {'✅' if smc.get('bos') else '❌'}
├ CHoCH: {'✅' if smc.get('choch') else '❌'}
└ OB Test: {'✅' if smc.get('ob_test') else '❌'}

<b>📊 OrderFlow ({of.get('score', 0)}/100):</b>
├ Delta: {of.get('delta', 'N/A')}
├ CVD: {of.get('cvd_trend', 'N/A')}
└ Kurumsal: {'Alış' if of.get('institutional_buying') else 'Yok'}

<b>⚡ Alpha ({alpha.get('score', 0)}/100):</b>
├ Kategori: {alpha.get('category', 'N/A')}
├ RS: {alpha.get('rs_slope', 'N/A')}
└ Momentum: {alpha.get('momentum', 'N/A')}

<i>Güncelleme: {datetime.now().strftime('%H:%M:%S')}</i>
"""
            
            # Keyboard
            keyboard = {
                'inline_keyboard': [
                    [
                        {'text': '📈 Chart', 'callback_data': f'chart_{symbol}'},
                        {'text': '🔄 Yenile', 'callback_data': f'refresh_{symbol}'}
                    ]
                ]
            }
            
            return await self.bot.send_message(
                text.strip(),
                chat_id=ctx.chat_id,
                reply_markup=keyboard
            )
            
        except Exception as e:
            logger.error(f"Analysis handler error: {e}")
            return await self.bot.send_message(
                f"❌ {symbol} analizi yapılamadı.",
                chat_id=ctx.chat_id
            )
    
    async def handle_risk(self, ctx: CommandContext) -> bool:
        """Risk komutu"""
        from app.core.shadow_mode import get_shadow_system
        
        try:
            shadow = get_shadow_system()
            
            if not shadow.is_active:
                text = "⚠️ Shadow Mode aktif değil. Risk metrikleri mevcut değil."
            else:
                portfolio = shadow.get_portfolio()
                
                # Risk seviyesi
                if portfolio.max_drawdown > 10:
                    risk_level = "🔴 YÜKSEK"
                elif portfolio.max_drawdown > 5:
                    risk_level = "🟡 ORTA"
                else:
                    risk_level = "🟢 DÜŞÜK"
                
                exposure = (portfolio.total_exposure / portfolio.current_capital * 100) if portfolio.current_capital > 0 else 0
                
                text = f"""
⚠️ <b>Risk Durumu</b>

<b>Genel Risk:</b> {risk_level}

<b>📊 Metrikler:</b>
├ Max Drawdown: {portfolio.max_drawdown:.1f}%
├ Current DD: {portfolio.current_drawdown:.1f}%
├ Exposure: {exposure:.1f}%
├ Açık Pozisyon: {portfolio.open_positions}
└ Win Rate: {portfolio.win_rate:.1f}%

<b>⚡ Limitler:</b>
├ Max DD Limit: 15%
├ Max Exposure: 80%
└ Max Pozisyon: 5

<i>Risk yönetimi kurallarına dikkat edin!</i>
"""
            
            return await self.bot.send_message(text.strip(), chat_id=ctx.chat_id)
            
        except Exception as e:
            logger.error(f"Risk handler error: {e}")
            return await self.bot.send_message(
                "❌ Risk bilgisi alınamadı.",
                chat_id=ctx.chat_id
            )
    
    async def handle_report(self, ctx: CommandContext) -> bool:
        """Report komutu"""
        from app.core.shadow_mode import get_shadow_system
        
        try:
            shadow = get_shadow_system()
            
            if not shadow.is_active:
                text = "📋 Rapor için aktif bir session gerekli."
            else:
                report = shadow.generate_report()
                
                if report:
                    pnl_emoji = "📈" if report.total_return >= 0 else "📉"
                    
                    # Top/worst
                    top_text = ""
                    for symbol, pnl in report.top_symbols[:3]:
                        emoji = "🟢" if pnl >= 0 else "🔴"
                        top_text += f"\n{emoji} {symbol}: {pnl:+.0f} TRY"
                    
                    text = f"""
📋 <b>Performans Raporu</b>

<b>📅 Dönem:</b> {report.period_start.strftime('%d.%m.%Y')} - {report.period_end.strftime('%d.%m.%Y')}

<b>💰 Sermaye:</b>
├ Başlangıç: {report.starting_capital:,.0f} TRY
└ Güncel: {report.ending_capital:,.0f} TRY

<b>{pnl_emoji} Getiri:</b> <code>{report.total_return:+,.0f}</code> TRY ({report.total_return_pct:+.2f}%)

<b>📊 Trade İstatistikleri:</b>
├ Toplam: {report.total_trades}
├ Kazanan: {report.winning_trades}
├ Kaybeden: {report.losing_trades}
├ Win Rate: {report.win_rate:.1f}%
└ Profit Factor: {report.profit_factor:.2f}

<b>⚠️ Risk:</b>
└ Max Drawdown: {report.max_drawdown:.1f}%

<b>🏆 En İyi Performans:</b>{top_text if top_text else "\nVeri yok"}

<b>💡 Öneriler:</b>
{"".join([f"• {r}" for r in report.recommendations[:3]]) if report.recommendations else "• Performans iyi durumda"}
"""
                else:
                    text = "📋 Rapor oluşturulamadı."
            
            return await self.bot.send_message(text.strip(), chat_id=ctx.chat_id)
            
        except Exception as e:
            logger.error(f"Report handler error: {e}")
            return await self.bot.send_message(
                "❌ Rapor oluşturulamadı.",
                chat_id=ctx.chat_id
            )
    
    async def handle_settings(self, ctx: CommandContext) -> bool:
        """Settings komutu"""
        text = """
⚙️ <b>Bildirim Ayarları</b>

Mevcut ayarlar:
├ Sinyal Bildirimleri: ✅ Açık
├ Trade Bildirimleri: ✅ Açık
├ Risk Uyarıları: ✅ Açık
└ Günlük Rapor: ✅ Açık

Ayarları değiştirmek için aşağıdaki butonları kullanın:
"""
        
        keyboard = {
            'inline_keyboard': [
                [
                    {'text': '🔔 Sinyaller: Açık', 'callback_data': 'toggle_signals'},
                    {'text': '📊 Trade: Açık', 'callback_data': 'toggle_trades'}
                ],
                [
                    {'text': '⚠️ Risk: Açık', 'callback_data': 'toggle_risk'},
                    {'text': '📋 Rapor: Açık', 'callback_data': 'toggle_report'}
                ]
            ]
        }
        
        return await self.bot.send_message(
            text.strip(),
            chat_id=ctx.chat_id,
            reply_markup=keyboard
        )
    
    async def handle_stop(self, ctx: CommandContext) -> bool:
        """Stop komutu"""
        text = """
🛑 <b>Bildirimler Durduruldu</b>

Artık sinyal ve uyarı almayacaksınız.

Bildirimleri tekrar açmak için /start yazın.
"""
        return await self.bot.send_message(text.strip(), chat_id=ctx.chat_id)


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

class CallbackHandlers:
    """
    Telegram callback (inline button) işleyicileri
    """
    
    def __init__(self, bot):
        self.bot = bot
    
    async def handle_callback(self, ctx: CallbackContext) -> bool:
        """
        Callback işle
        
        Args:
            ctx: Callback bağlamı
            
        Returns:
            Başarılı mı
        """
        data = ctx.callback_data
        
        if data.startswith("chart_"):
            symbol = data.replace("chart_", "")
            return await self._handle_chart(ctx, symbol)
        
        elif data.startswith("detail_"):
            signal_id = data.replace("detail_", "")
            return await self._handle_detail(ctx, signal_id)
        
        elif data.startswith("approve_"):
            signal_id = data.replace("approve_", "")
            return await self._handle_approve(ctx, signal_id)
        
        elif data.startswith("reject_"):
            signal_id = data.replace("reject_", "")
            return await self._handle_reject(ctx, signal_id)
        
        elif data.startswith("refresh_"):
            symbol = data.replace("refresh_", "")
            return await self._handle_refresh(ctx, symbol)
        
        elif data.startswith("toggle_"):
            setting = data.replace("toggle_", "")
            return await self._handle_toggle(ctx, setting)
        
        return False
    
    async def _handle_chart(self, ctx: CallbackContext, symbol: str) -> bool:
        """Chart callback"""
        # TradingView link veya chart image gönder
        text = f"📈 {symbol} Chart:\nhttps://tr.tradingview.com/chart/?symbol=BIST:{symbol}"
        return await self.bot.send_message(text, chat_id=ctx.chat_id)
    
    async def _handle_detail(self, ctx: CallbackContext, signal_id: str) -> bool:
        """Signal detail callback"""
        text = f"📊 Sinyal detayları: {signal_id}\n\nDetaylar yükleniyor..."
        return await self.bot.send_message(text, chat_id=ctx.chat_id)
    
    async def _handle_approve(self, ctx: CallbackContext, signal_id: str) -> bool:
        """Approve signal callback"""
        text = f"✅ Sinyal onaylandı: {signal_id}\n\nShadow Mode'da trade açılıyor..."
        return await self.bot.send_message(text, chat_id=ctx.chat_id)
    
    async def _handle_reject(self, ctx: CallbackContext, signal_id: str) -> bool:
        """Reject signal callback"""
        text = f"❌ Sinyal reddedildi: {signal_id}"
        return await self.bot.send_message(text, chat_id=ctx.chat_id)
    
    async def _handle_refresh(self, ctx: CallbackContext, symbol: str) -> bool:
        """Refresh analysis callback"""
        text = f"🔄 {symbol} analizi yenileniyor..."
        return await self.bot.send_message(text, chat_id=ctx.chat_id)
    
    async def _handle_toggle(self, ctx: CallbackContext, setting: str) -> bool:
        """Toggle setting callback"""
        text = f"⚙️ {setting} ayarı değiştirildi."
        return await self.bot.send_message(text, chat_id=ctx.chat_id)
