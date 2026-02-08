#!/bin/bash
curl -s 'https://fapi.binance.com/fapi/v1/ticker/24hr' | \
jq -r '.[] | select(.symbol | endswith("USDT")) | select(.symbol | test("^(BTC|ETH|SOL|BNB|XRP|ADA|AVAX|DOT|MATIC|LINK|UNI|ATOM|LTC|APT|ARB|OP|IMX|NEAR|FET|INJ|SUI|TAO|AAVE|RNDR)")) | [.symbol, (.quoteVolume | tonumber)] | @tsv' | \
sort -t$'\t' -k2 -rn | head -12 | awk '{printf "%2d. %-12s $%.2fB\n", NR, $1, $2/1000000000}'

echo ""
echo "Comma-separated:"
curl -s 'https://fapi.binance.com/fapi/v1/ticker/24hr' | \
jq -r '.[] | select(.symbol | endswith("USDT")) | select(.symbol | test("^(BTC|ETH|SOL|BNB|XRP|ADA|AVAX|DOT|MATIC|LINK|UNI|ATOM|LTC|APT|ARB|OP|IMX|NEAR|FET|INJ|SUI|TAO|AAVE|RNDR)")) | [.symbol, (.quoteVolume | tonumber)] | @tsv' | \
sort -t$'\t' -k2 -rn | head -12 | awk '{print $1}' | paste -sd,
