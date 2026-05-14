import random
from config import stock_analyzer


class PortfolioBuilder:
    def __init__(self, stock_analyzer_instance):
        self.stock_analyzer = stock_analyzer_instance
        self.average_annual_returns = {
            "Large Cap": 0.12,
            "Mid Cap": 0.15,
            "Small Cap": 0.18,
        }
        self.risk_profiles = {
            "Conservative": {"Large Cap": 0.50, "Mid Cap": 0.30, "Small Cap": 0.20},
            "Moderate": {"Large Cap": 0.30, "Mid Cap": 0.50, "Small Cap": 0.20},
            "Aggressive": {"Large Cap": 0.20, "Mid Cap": 0.40, "Small Cap": 0.40},
        }

    def get_asset_allocation(self, risk_profile):
        return self.risk_profiles.get(risk_profile, {})

    def project_investment(
        self, initial_investment, monthly_sip, duration_years, risk_profile
    ):
        allocation = self.get_asset_allocation(risk_profile)
        total_projected_value = 0

        for asset, percentage in allocation.items():
            rate = self.average_annual_returns.get(asset, 0)
            fv_lump_sum = initial_investment * percentage * ((1 + rate) ** duration_years)

            monthly_rate = rate / 12
            duration_months = duration_years * 12
            if monthly_rate != 0:
                fv_sip = (
                    monthly_sip
                    * percentage
                    * (((1 + monthly_rate) ** duration_months - 1) / monthly_rate)
                )
            else:
                fv_sip = monthly_sip * percentage * duration_months

            total_projected_value += fv_lump_sum + fv_sip

        return total_projected_value

    def get_stock_suggestions(self, risk_profile):
        allocation = self.get_asset_allocation(risk_profile)
        suggestions = {}
        for asset, percentage in allocation.items():
            if percentage > 0:
                if asset == "Large Cap":
                    available_stocks = self.stock_analyzer.large_cap_stocks
                elif asset == "Mid Cap":
                    available_stocks = self.stock_analyzer.mid_cap_stocks
                elif asset == "Small Cap":
                    available_stocks = self.stock_analyzer.small_cap_stocks
                else:
                    available_stocks = []

                if available_stocks:
                    num_suggestions = min(3, len(available_stocks))
                    suggestions[asset] = random.sample(available_stocks, num_suggestions)
        return suggestions
