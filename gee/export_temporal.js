var study_area = ee.Geometry.Rectangle([109.71, 28.89, 111.30, 29.80]);
Map.centerObject(study_area, 9);

var S2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(study_area)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40));

var CHIRPS = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterBounds(study_area);

function maskS2(img) {
  var qa = img.select('QA60');
  var mask = qa.bitwiseAnd(1 << 10).eq(0).and(qa.bitwiseAnd(1 << 11).eq(0));
  return img.updateMask(mask)
    .select(['B2','B3','B4','B8','B11','B12'])
    .rename(['BLUE','GREEN','RED','NIR','SWIR1','SWIR2'])
    .divide(10000)
    .copyProperties(img, ['system:time_start']);
}

var S2_clean = S2.map(maskS2);
var S2_fallback = S2_clean.filterDate('2018-01-01','2026-06-30').median();

function safeS2(dateStr, months) {
  var start = ee.Date(dateStr);
  var end = start.advance(months, 'month');
  var col = S2_clean.filterDate(start, end);
  return ee.Image(ee.Algorithms.If(col.size().gt(0), col.median(), S2_fallback));
}

function safeRain(dateStr, months) {
  var start = ee.Date(dateStr);
  var end = start.advance(months, 'month');
  return CHIRPS.filterDate(start, end).sum().rename('RAIN').clamp(0, 1500);
}

function getNDVI(img) {
  return img.normalizedDifference(['NIR','RED']).clamp(-1, 1);
}

function getNBR(img) {
  return img.normalizedDifference(['NIR','SWIR2']).clamp(-1, 1);
}

function getBSI(img) {
  return img.expression(
    '((SWIR+RED)-(NIR+BLUE))/((SWIR+RED)+(NIR+BLUE)+0.001)',
    {'SWIR': img.select('SWIR1'), 'RED': img.select('RED'),
     'NIR': img.select('NIR'), 'BLUE': img.select('BLUE')}
  ).clamp(-1, 1);
}

function getSWIR(img) {
  return img.select('SWIR1').clamp(0, 0.8);
}

function computeTemporal(year, month) {
  var tgt = ee.Date.fromYMD(year, month, 1);
  var tgtStr = tgt.format('YYYY-MM-dd').getInfo();
  
  var win12 = S2_clean.filterDate(tgt.advance(-12,'month'), tgt);
  var win6 = S2_clean.filterDate(tgt.advance(-6,'month'), tgt);
  
  var med12 = ee.Image(ee.Algorithms.If(win12.size().gt(0), win12.median(), S2_fallback));
  var med6 = ee.Image(ee.Algorithms.If(win6.size().gt(0), win6.median(), S2_fallback));
  
  var s2_cur = safeS2(tgtStr, 1);
  var s2_12mo = safeS2(tgt.advance(-12,'month').format('YYYY-MM-dd'), 1);
  
  var ndvi_cur = getNDVI(s2_cur);
  var ndvi_mean = getNDVI(med12);
  var ndvi_trend = ndvi_cur.subtract(getNDVI(s2_12mo));
  
  var nbr_cur = getNBR(s2_cur);
  var nbr_mean = getNBR(med12);
  
  var bsi_cur = getBSI(s2_cur);
  var bsi_mean = getBSI(med12);
  
  var swir_cur = getSWIR(s2_cur);
  var swir_mean = getSWIR(med12);
  
  var rain_cur = safeRain(tgtStr, 1);
  var rain_mean = safeRain(tgt.advance(-12,'month').format('YYYY-MM-dd'), 12).divide(12);
  var rain_3mo = safeRain(tgt.advance(-3,'month').format('YYYY-MM-dd'), 3);
  var rain_6mo = safeRain(tgt.advance(-6,'month').format('YYYY-MM-dd'), 6);
  
  var pre6_rain = safeRain(tgt.advance(-6,'month').format('YYYY-MM-dd'), 6);
  var post6_rain = safeRain(tgtStr, 6);
  var rain_change = post6_rain.subtract(pre6_rain);
  
  var s2_post6 = safeS2(tgtStr, 6);
  var ndvi_change = getNDVI(s2_post6).subtract(getNDVI(med6));
  
  var rain_score = rain_cur.divide(200).multiply(0.30)
    .add(rain_mean.divide(120).multiply(0.15))
    .add(rain_3mo.divide(400).multiply(0.08))
    .add(rain_6mo.divide(700).multiply(0.05))
    .add(rain_change.divide(400).multiply(0.04));
  
  var ndvi_score = ndvi_cur.add(1).divide(2).multiply(-0.09)
    .add(ndvi_mean.add(1).divide(2).multiply(-0.04))
    .add(ndvi_trend.multiply(-0.02))
    .add(ndvi_change.multiply(-0.02));
  
  var nbr_score = nbr_cur.add(1).divide(2).multiply(-0.04)
    .add(nbr_mean.add(1).divide(2).multiply(-0.02));
  
  var bsi_score = bsi_cur.add(1).divide(2).multiply(0.04)
    .add(bsi_mean.add(1).divide(2).multiply(0.02));
  
  var swir_score = swir_cur.multiply(0.04).add(swir_mean.multiply(0.02));
  
  var change_score = ndvi_change.multiply(-0.02).add(rain_change.divide(300).multiply(0.02));
  
  var score = rain_score.add(ndvi_score).add(nbr_score).add(bsi_score).add(swir_score).add(change_score);
  
  var k = 7;
  var susc = ee.Image(1).divide(ee.Image(1).add(score.multiply(-k).exp()))
    .rename('susceptibility').clamp(0, 1);
  
  return susc.clip(study_area);
}

function seasonMean(monthList, yrStart, yrEnd) {
  var imgs = [];
  for (var yr = yrStart; yr <= yrEnd; yr++) {
    for (var mi = 0; mi < monthList.length; mi++) {
      var mo = monthList[mi];
      if (yr === 2026 && mo > 6) continue;
      imgs.push(computeTemporal(yr, mo));
    }
  }
  return ee.ImageCollection(imgs).mean().rename('susceptibility').clip(study_area);
}

var annual = seasonMean([1,2,3,4,5,6,7,8,9,10,11,12], 2018, 2026);
var dry = seasonMean([11,12,1,2,3,4], 2018, 2026);
var rainy = seasonMean([5,6,7,8,9,10], 2018, 2026);
var sep2020 = computeTemporal(2020, 9);
var jun2022 = computeTemporal(2022, 6);
var aug2023 = computeTemporal(2023, 8);
var jul2024 = computeTemporal(2024, 7);
var jun2025 = computeTemporal(2025, 6);
var jun2026 = computeTemporal(2026, 6);

var FOLDER = 'LSM_Temporal_Trigger';
var SCALE = 30;
var CRS = 'EPSG:4326';

var names = ['Annual_Mean','Dry_Season','Rainy_Season','Sept_2020','June_2022',
             'Aug_2023','July_2024','June_2025','June_2026'];
var imgs = [annual,dry,rainy,sep2020,jun2022,aug2023,jul2024,jun2025,jun2026];

for (var n = 0; n < names.length; n++) {
  var nm = names[n];
  var img = imgs[n];
  
  Export.image.toDrive({
    image: img,
    description: 'temporal_trigger_' + nm,
    folder: FOLDER,
    fileNamePrefix: 'ZHJ_LSM_' + nm,
    scale: SCALE,
    region: study_area,
    crs: CRS,
    maxPixels: 1e10,
    fileFormat: 'GeoTIFF'
  });
}
