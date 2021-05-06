var file_dir = [];
var file_dir_orig = [];
var file_dir_reg = [];
var error = document.getElementById("warning_box")
var layer_ids = [];
var original_weights_identifier = "Oweights"
var reg_weights_identifier = "Rweights"
var csv_orig, csv_reg;
var global_data = [];
var orig_global_data = [];
var reg_global_data = [];
var reader = new FileReader();
var reader2 = new FileReader();
var reader3 = new FileReader();
var bin_size = 20;
var margin,width,height;
var data_header = [];
var data_header_reg = [];
var max_y = 0.0001;
var min_y =  -0.0001;
var orig_local_data = [];
var columns = [];
var x = d3.scaleLinear();
var y = d3.scaleLinear();
var orig_raw_data = [];
var reg_raw_data = [];
var layer_val ;
var hierarchy_level ;
var bin_level = "bin";
var info_model,info_layer,info_bins,info_bin_size, info_layer_hierarchy,info_ids, info_epochs ;
data_orig = [];
var  legacy_line_flag = 0 // flag to check if we need a parent bin/legacy line
var legacy_orig_arr = [] ;
var legacy_reg_arr = [];

var acc_png,loss_png,recons_png,code ;
var image=new Image();
var acc_img=new Image();
var loss_img =new Image();
var recons_img=new Image();
var code_text;


document.getElementById("filepicker").addEventListener("change", function(event) {
  let output = document.getElementById("listing");
  let files_obj = event.target.files;
  files = [].slice.call(files_obj).sort()
  files = files.sort((fileA, fileB) => fileA.name.localeCompare(fileB.name));


  for (let i=0; i<files.length; i++) {
    if(files[i].name.includes(original_weights_identifier)){
//        let item = document.createElement("li");
//        item.innerHTML = files[i].webkitRelativePath;
        file_dir.push(files[i])
        file_dir_orig.push(files[i]);
     }
     else if (files[i].name.includes(reg_weights_identifier)){
        file_dir.push(files[i])
        file_dir_reg.push(files[i])
        var id = (files[i].name).substring(8,10)
        if (id.includes(".")){
            id = "All_Layers"
        }
        layer_ids.push(id);
        let item = document.createElement("li");
        //item.innerHTML =id
        //output.appendChild(item);
     }
    else if  (files[i].name.includes("py")) {
            code = files[i]
        loadFile_code(code)
        console.log(code)
    }
     else if  (files[i].name.includes("png")) {
            if(files[i].name.includes("Acc.png")){
                acc_png = files[i]
                var reader = new FileReader();
                reader.readAsDataURL(acc_png);
                reader.onload = function (e) {
                    // var image=new Image();
                    acc_img.src=e.target.result;
                }
            }
            else if(files[i].name.includes("Loss.png")){
                loss_png = files[i]
                var reader = new FileReader();
                reader.readAsDataURL(loss_png);
                reader.onload = function (e) {
                    // var image=new Image();
                    loss_img.src=e.target.result;
                }

            }
            else if(files[i].name.includes("ae-images.png")){
                recons_png = files[i]
                var reader = new FileReader();
                reader.readAsDataURL(recons_png);
                reader.onload = function (e) {
                    // var image=new Image();
                    recons_img.src=e.target.result;
                }

            }

    }


//        output.appendChild(item);
    if (i == (files.length -1)){
        if((file_dir.length%2) != 0 ){error.innerHTML = "Weight file is not a factor of 2 i.e no Reg file for every Orig " }
    }
  };

    function loadFile_code(file) {
        reader3.addEventListener("load", parseFile_code, false);
        if (file) {
            reader3.readAsText(file);
        }
    }
//Parse Read file
    function parseFile_code(){
        code_text = reader3.result;

    }












    // Fil drop down values now
    var selectList = document.getElementById("layerSelect");
    //Create and append the options
    for (var i = 0; i < layer_ids.length; i++) {
        var option = document.createElement("option");
        option.value = i;
        option.text = layer_ids[i];
        selectList.appendChild(option);
    }

}, false);



function selectClicked(selectObject) {
  layer_val= selectObject.value;
  //if (value == "All_Layers"){ value = []}
  csv_orig = file_dir_orig[layer_val]
  csv_reg =  file_dir_reg[layer_val]
    info_layer = layer_ids[layer_val]
    hierarchy_level = "Layer " + info_layer;
  info_layer_hierarchy = hierarchy_level

  loadFile_orig(csv_orig)
  loadFile_reg(csv_reg)

}


function loadFile_orig(file1) {
  reader.addEventListener("load", parseFile_orig, false);
  if (file1) {
    reader.readAsText(file1);
    console.log("file name",file1)
    // console.log("Information: Layer --> " , file1.name)
    // console.log("Information: Data Set --> " , file1.webkitRelativePath.split('/')[0])
    info_model =   file1.webkitRelativePath.split('/')[0]
  }
}
//Parse Read file
function parseFile_orig(){
  var lines = reader.result.split('\n');
  var parsed = d3.csvParse(reader.result);
  orig_raw_data = parsed;
  // only useful when csv done have "Epoch:" in header names
  // orig_local_data = balanceEpochs(orig_raw_data)
  orig_local_data = orig_raw_data
}



function loadFile_reg(file2) {
  reader2.addEventListener("load", parseFile_reg, false);
  if (file2) {
    reader2.readAsText(file2);
    console.log("file name",file2)

  }
}

//Parse Read file
function parseFile_reg(){
  var lines = reader2.result.split('\n');
  var parsed = d3.csvParse(reader2.result);
  reg_raw_data = parsed;
 // only useful when csv done have "Epoch:" in header names
//  reg_global_data = balanceEpochs(reg_raw_data)

  reg_global_data = reg_raw_data
  formSkelton();
}



function randomColor()
{
     color='rgb('+Math.round(Math.random()*255)+','+Math.round(Math.random()*255)+','+Math.round(Math.random()*255)+')';
     return color;
}

function formSkelton(){
       d3.select("svg").remove();
        data_header = orig_local_data.columns

        data_header_reg  = reg_global_data.columns
        var w = window.innerWidth;
        margin = {top: 40, right: 60, bottom: 30, left: 60},
        width = w - margin.left - margin.right-20,
        height = 400 - margin.top - margin.bottom;

        // append the svg object to the body of the page
         svg = d3.select("#my_dataviz")
          .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom +20)
          .append("g")  //wht is g --> http://tutorials.jenkov.com/svg/g-element.html
            .attr("transform",
                  "translate(" + margin.left + "," + margin.top + ")");


        //data = orig_global_data // considering reg and orig have same row, columns size
        if (orig_local_data.length > 0)
        columns = Object.keys( orig_local_data[0] )
        else
        columns = Object.keys( reg_global_data[0] )

        processLines();


}

function processLines(){
    max_y = 0.0001;
    min_y =  -0.0001;


    if (orig_local_data.length < orig_raw_data.length)
        info_ids = orig_local_data.length + " of  "  + orig_raw_data.length
    else
        info_ids = orig_local_data.length
    info_epochs = (data_header.length-1)


    numberOfColumns = columns.length -1;

    if (orig_local_data.length < bin_size){ bin_size = orig_local_data.length}
    else (bin_size = 20)
    info_bins = bin_size

        sort_by_col = "Epoch:"+(numberOfColumns-1)

//sorting orig data
        orig_local_data.sort(function(a, b) {
            return parseFloat(a[sort_by_col]) - parseFloat(b[sort_by_col]);
        });

        data_trans = transpose(getAverage(orig_local_data,bin_size))
        data_trans.length = data_header.length - 1 // avoid ids to be printed

        data_trans_reg = transpose(filterDataForIds(data_trans,reg_global_data))
        data_trans_reg.length = data_header_reg.length -1
        // d3.selectAll(".legacy").remove()
        legacy_line_flag = 0
        drawLine(data_trans,data_trans_reg)

}

function transpose(data) {
  let result = {};
  for (let row of data) {
    for (let [key, value] of Object.entries(row)) {
      if (key.includes("Epoch:")) {key = key.replace("Epoch:",''); }
      result[key] = result[key] || [];
      result[key].push(value);
      if(key != "Id" && (value> 5 || value < -5  || typeof(value) !== 'number' )){
      // console.log("Bakchodi value alert");
      }

      if(key != "Id" && value> max_y){max_y = value;}
      else if ( key != "Id" && value < min_y){min_y= value;}
    }
  }
  return result;
}



function getAverage(data,bin_size){

            var bin_space =  Math.floor(data.length/bin_size)//no of ids in each bin
            var diff = data.length - (bin_space*(bin_size-1));
            var avg_data =  [];
            // for now todo - remove it
            // data_header = data_header_reg
            var temp_data = new Array(data_header.length); // next shal rather be ket value with keys being epoch
            var ids = [];
            var rows_arr = [];
            console.log("Data length is "+data.length)
            for(var iRow=0; iRow<data.length ;++iRow){
                rows_arr.push(data[iRow])
                for (var iCol =0 ; iCol<data_header.length ;  ++iCol){
                    if(data_header[iCol] == "Id"){
                        ids.push(parseFloat(data[iRow][data_header[iCol]]))
                        }
                    else{
                        if(temp_data[data_header[iCol]]== undefined ){
                            temp_data[data_header[iCol]] =  (parseFloat(data[iRow][data_header[iCol]]))


                        }
                        else{
                            temp_data[data_header[iCol]] =  temp_data[data_header[iCol]] + (parseFloat(data[iRow][data_header[iCol]]))

                            //next count mean
                        }
                    }
                    // push all remainig data in last bin
                    if (avg_data.length == (bin_size -1)) {
                    bin_space = 10000000
                    } // this iwll render the bin space condition in next line as always false
                    // to push tem data to avergae array, when its on last col of row where bin range is over
                    if ((iCol == (data_header.length -1) )&& (((iRow)%bin_space)==0 || ((iRow+1)==data.length) || (avg_data.length == 0 && (iRow +1)%bin_space == 0 )) && (iRow!= 0 || bin_space ==1)){

                        //making avergae from sum
                        for(var iTemp = 0; iTemp<data_header.length; ++iTemp){
                            if(data_header[iTemp] != "Id"){
                                temp_data[data_header[iTemp]] = temp_data[data_header[iTemp]] /(ids.length)
                            }

                        }
                        temp_data["Id"]= ids; //feeding ids to temp data as ids are always first and temp data [0] was 0 , see if block 7 lines above
                        temp_data["Sd_min"] = getSD(rows_arr).epoch_sd_min
                        temp_data["Sd_max"] = getSD(rows_arr).epoch_sd_max
                        avg_data.push(temp_data)

                        temp_data=[];
                        ids = [];
                        rows_arr  = [];
                    }
                }

            }
            return avg_data
}


function filterDataForIds(orig_binned_data, reg_raw_data){
//inputs - 1) orig data (sorted and comprssed in bins  ) , 2) raw reg data
// processing , make same bins with same ids as that of orig data
//output --> reg bins data with same bins and same ids in each bin as that of reg data
var ids = [];
var reg_binned_data = [];
    for(var i = 0 ;i < bin_size ; ++i ){
        var temp = [];
        var temp_data = []
        var rows_arr = [];
    // assuming length don't cover the keys as we have fed the length manually
        temp_data["Id"] = orig_binned_data.Id[i]
        for (var i2 = 0 ;i2<(orig_binned_data.Id[i].length);++i2){
            var temp2 =(reg_global_data.filter(function (el) {return el.Id == temp_data["Id"][i2]}))[0]
            rows_arr.push(temp2)
            for (var cols = 1 ;cols< (data_header_reg.length) ;cols++ ){
                    if(temp_data[data_header_reg[cols]]== undefined ){temp_data[data_header_reg[cols]] = parseFloat(temp2[data_header_reg[cols]])/(orig_binned_data.Id[i].length)}
                    else{temp_data[data_header_reg[cols]] += parseFloat(temp2[data_header_reg[cols]])/(orig_binned_data.Id[i].length)}
                }
        }
        temp_data["Sd_min"] = getSD(rows_arr).epoch_sd_min
        temp_data["Sd_max"] = getSD(rows_arr).epoch_sd_max
        reg_binned_data.push(temp_data)
        temp_data = []
        rows_arr  = [];



   }
   return reg_binned_data; //(7)
}


function drawLine(data_orig,data_reg){
    /**
     * Inputs - Original data, Reg model data
     *
     */

    updateInfoBox()
        d3.selectAll('.line').remove()
        d3.selectAll('.legacy').remove()



        x.domain([0,(data_orig.length-1)])
          .range([ 0, width ]);

        xAxis = svg.append("g").attr("class", "xaxis").attr("stroke", "#868787").style("font", "14px times").attr("transform", "translate(0," + height + ")").text("income per capita, inflation-adjusted (dollars)")
                    .call(d3.axisBottom(x))
        d3.selectAll('.yaxis').remove()
        y.domain([min_y,max_y]).range([ height, 0 ]);
        yAxis = svg.append("g").attr("stroke", "#868787").attr("class", "yaxis").style("font", "14px times").call(d3.axisLeft(y))


        //todo somehitng here is causing double digits on y axis


        var clip = svg.append("defs").append("svg:clipPath")
                    .attr("id", "clip")
                    .append("svg:rect")
                    .attr("width", width )
                    .attr("height", height )
                    .attr("x", 0)
                    .attr("y", 0);





    for(var k=0;k<bin_size;k++){
        var line = svg.append('g')
                       .attr("clip-path", "url(#clip)")

        var sd_min = [];
        var sd_max = [];
        var sd_data_rows = [];
        var random_color = randomColor()
        var classk = "line " + "class"+k; // will asisgn two classes o each line 1) line 2) uniqute i.e  class + k
        //Draw Orig lines
        if (legacy_line_flag){
            d3.selectAll(".legacy").remove() // to remove old legacy lines i.e from last bins
            line.append("path")
                .datum(legacy_orig_arr) // 10. Binds data to the line
                .attr("class", "legacy")
                .attr("fill", "none")
                .attr("stroke", random_color)
                .attr("stroke-width", 22.5)
                .style('opacity', 0.40)
                .attr('id',  + k)
                .attr("d", d3.line()
                    .x(function(d,i) {
                        //console.log("Crap")
                        return x(i)
                    })
                    .y(function(d) {return y(d) })

                )

            line.append("path")
                .datum(legacy_reg_arr) // 10. Binds data to the line
                .attr("class", "legacy")
                .attr("fill", "none")
                .attr("stroke", random_color)
                .style("stroke-dasharray", ("3, 3"))
                .attr("stroke-width", 22.5)
                .style('opacity', 0.40)
                .attr('id',  + k)
                .attr("d", d3.line()
                    .x(function(d,i) {
                        //console.log("Crap")
                        return x(i)
                    })
                    .y(function(d) {return y(d) })

                )


            legacy_line_flag=0
        }

        line_orig = line.append("path")
                    .datum(data_orig) // 10. Binds data to the line
                    .attr("class", classk)
                    .attr("fill", "none")
                    .attr("stroke", random_color)
                    .attr("stroke-width", 1.5)
                    .style('opacity', 0.75)
                    .attr('id',  + k)
                    .attr("d", d3.line()
                                  .x(function(d,i) {
                                      //console.log("Crap")
                                      return x(i)
                                  })
                                  .y(function(d) {return y(d[k]) })

                    )

                    .on("mouseover", function(d,line) {
                       d3.selectAll('.line')
                            .style('opacity', 0.1);
                       d3.selectAll( "."+ "class" + this.id)
                              .style('opacity', 0.75)
                              .style("stroke-width", 3.5)
                              .style("cursor", "pointer")

                    })


                    .on("mouseout", function(d) {
                    d3.selectAll('.line')
                        .style('opacity', 0.75);})


                    .on("click", function(d,line) {
                        d3.selectAll('.area').remove();
                        //if totatl data is less then 40 , and lines are 20 , there is almost no SD to show
                        if(orig_local_data.length > 20) {
                            draw_sd_zussamen(data_trans, data_trans_reg, this.id)
                        }
                        d3.selectAll( "."+ "class" + this.id)
                                .attr("class", "survivor")
                                // .style("stroke-width", 12.75)
                                //  .style('opacity', 0.20)

                    })


        //Drawing Reg Lines , dashed
        indi_line_reg = line.append("path")
                    .datum(data_reg) // 10. Binds data to the line
                    .attr("class", classk)
                    .attr("fill", "none")
                    .attr("stroke", random_color)
                    .style("stroke-dasharray", ("3, 3"))
                    .attr("stroke-width", 1.5)
                    .style('opacity', 0.75)
                    .attr("d", d3.line()
                                  .x(function(d,i) {return x(i)})
                                  .y(function(d) {return y(d[k])})

                    )

                    .attr('id', + k)

                    .on("mouseover", function(d) {
                        d3.selectAll('.line')
                              .style('opacity', 0.1);
                        d3.selectAll( "."+ "class" + this.id)
                            .style('opacity', 0.75)
                            .style("stroke-width", 3.5)
                            .style("cursor", "pointer")

                    })
                  .on("mouseout", function(d) {
                      d3.selectAll('.line')
                          .style('opacity', 0.75);})





        function draw_sd_zussamen(data_orig,data_reg,id){
            //Red for Orig
            //Steel blue for Reg
            //Yellow for common
            extract_data_for_legacy_line(id) // feeind arrays to redraw legacy line
            legacy_line_flag = 1
            var sd_data =[];
            sd_data[0]=sd_min= Object.values(data_orig["Sd_min"][id])
            sd_data[1]=sd_max = Object.values(data_orig["Sd_max"][id])
            sd_data_rows = transpose(sd_data)
            sd_data_rows["ids"]=data_orig.Id[id]
            sd_data_rows.length = sd_min.length
            var class_area = "area " + "orig_area";

            indi_line_sd =line.append("path")
                .datum(sd_data_rows)
                .attr("class", class_area)
                .attr("fill","rgb(249,111,111)")
                .style("opacity",0.4)
                .attr("d", d3.area()
                    .x(function(d,i) {return x(i); })
                    .y0(function(d,i) {return y(d[0]); })
                    .y1(function(d,i){return y(d[1]); })
                )
                .on("click", function(d) {
                    hierarchy_level = hierarchy_level + " " + bin_level  + " " +  id + "/" + bin_size;
                    bin_level = "sub"+bin_level
                    if(d["ids"].length > 1) {
                        filterData(d["ids"])
                    }
                    // store data for legacy lines(orig_reg) here
                });


            var sd_data_reg =[];
            sd_data[0]=reg_sd_min= Object.values(data_reg["Sd_min"][id])
            sd_data[1]=reg_sd_max = Object.values(data_reg["Sd_max"][id])
            sd_data_rows_reg = transpose(sd_data)
            sd_data_rows_reg["ids"]=data_orig.Id[id]
            sd_data_rows_reg.length = reg_sd_min.length
            class_area = "area " + "reg_area";

            indi_line_sd =line.append("path")
                .datum(sd_data_rows_reg)
                .attr("class", class_area)
                .attr("fill","lightsteelblue")
                .style("opacity",0.2)
                .attr("d", d3.area()
                    .x(function(d,i) {return x(i); })
                    .y0(function(d,i) {return y(d[0]); })
                    .y1(function(d,i){return y(d[1]); })
                )
                .on("click", function(d) {
                    hierarchy_level = hierarchy_level + " " + bin_level  + " " +  id + "/" + bin_size;
                    bin_level = "sub"+bin_level
                    if(d["ids"].length > 1) {
                        filterData(d["ids"])
                    }
                });



            // Draw diffrent color for common area to highlight the non common area by orig colors
            sd_common_min = [];
            sd_common_max =[];
            for (var i = 0 ;i< reg_sd_max.length;i++){
                (sd_min[i] < reg_sd_min[i] ? sd_common_min[i]= reg_sd_min[i] :  sd_common_min[i]= sd_min[i] );
                (sd_max[i] < reg_sd_max[i] ? sd_common_max[i]= sd_max[i] :  sd_common_max[i]= reg_sd_max[i] );
            }
            var sd_data_common =[];
            sd_data_common[0] = sd_common_min;
            sd_data_common[1] = sd_common_max;
            sd_data_common = transpose(sd_data_common)
            sd_data_common["ids"]=data_orig.Id[id]
            sd_data_common.length = sd_common_min.length
            class_area = "area " + "common_area";

            sd_common_area =line.append("path")
                .datum(sd_data_common)
                .attr("class", class_area)
                .attr("fill","yellow")
                .style("opacity",0.8)
                .attr("d", d3.area()
                    .x(function(d,i) {return x(i); })
                    .y0(function(d,i) {return y(d[0]); })
                    .y1(function(d,i){return y(d[1]); })
                )
                .on("click", function(d) {
                    hierarchy_level = hierarchy_level + " " + bin_level  + " " +  id + "/" + bin_size;
                    bin_level = "sub"+bin_level
                    if(d["ids"].length > 1) {
                        filterData(d["ids"])
                    }
                });


            window.setTimeout(function() {
                d3.selectAll('.area').remove();
            }, 2000)
        }



    }
}


function filterData(temp_rowids){
    d3.selectAll('.area').remove();
    d3.selectAll('.survivor').remove(); //so that the bin that was expanded last time , the main line shall be removed too
    console.log("Zooming on line click")
    rowids = temp_rowids
    var filterdData = [];
    if (rowids !== undefined && rowids.length != 0){
        for (i=0; i< orig_local_data.length; i++)
        {
            if (rowids.includes(parseInt(orig_local_data[i]["Id"])))
            {
                filterdData.push(orig_local_data[i])
            }
        }
    orig_local_data = filterdData ;
    }
    processLines();



}
// add epoch to header if not there
 function balanceEpochs(data){
    if(data.columns[1].includes("Epoch:")){return data}

    var balanced_data= [];
    var temp_data= []
    for (var i = 0 ; i < data.length ;i++){

         for (let [key, value] of Object.entries(data[i])) {
            if(key != "Id" && !key.includes("Epoch:")){
                temp_data["Epoch:"+key] = value
            }
         }
    temp_data["Id"] = data[i]["Id"]
    balanced_data.push(temp_data)
    temp_data = []
    }

    // to add epoch to cols and feed to balanced data
    for (var i =0;i<data.columns.length;i++){
        if(data.columns[i] != "Id" && data.columns[i] != "Ids") {
            data.columns[i]= "Epoch:"+data.columns[i]
        }
    }
   balanced_data["columns"]= data.columns

   return balanced_data;
 }

 function getSD(data){
    var epoch_arr=[]
    var epoch_mean = []
    var epoch_sd = [];
     var epoch_sd_min = [];
     var epoch_sd_max = [];
    var ids =[];

     let getMean = function (data) {
         return data.reduce(function (a, b) {
             return Number(a) + Number(b);
         }) / data.length;
     };

     let calcSD = function (data,m) {
         //let m = getMean(data);
         return Math.sqrt(data.reduce(function (sq, n) {
             return sq + Math.pow(n - m, 2);
         }, 0) / (data.length - 1));
     };

    for (var iCol=0;iCol<data_header.length;iCol++){
        if(data_header[iCol]!= "Id"){
            epoch_arr= data.map(x => x[data_header[iCol]])
            //epoch_mean[data_header[iCol]] = getMean(epoch_arr)
            // epoch_sd[data_header[iCol]] = math.std(epoch_arr)
            // calc SD and add to mean to find raneg of area
            epoch_sd_min[data_header[iCol]] = getMean(epoch_arr) - math.std(epoch_arr)
            epoch_sd_max[data_header[iCol]] = getMean(epoch_arr) + math.std(epoch_arr)

        }
        else{ ids = data.map(x => x["Id"]) }
    }


     return {ids,epoch_mean,epoch_sd_min,epoch_sd_max};

}

function updateInfoBox(){
    document.getElementById("info_model_name").innerHTML =  "Model - " + info_model
    document.getElementById("info_layer_name").innerHTML =  "Layer - " + info_layer
    document.getElementById("info_ids").innerHTML =  "Ids - " + info_ids
    document.getElementById("info_epochs").innerHTML =  "epochs - " + info_epochs
    document.getElementById("info_bins").innerHTML =  "Number of Bins - " + info_bins
    document.getElementById("info_layer_hierarchy").innerHTML =  "Layer Hierarchy-- " + hierarchy_level


}

function extract_data_for_legacy_line(id){
    legacy_orig_arr = [] ;
    legacy_reg_arr = [];
    for (var i = 0 ; i < data_header.length-1; i ++){
        legacy_orig_arr.push(data_trans[i][id])

    }
    for (var i = 0 ; i < data_header.length-1; i ++){
        legacy_reg_arr.push(data_trans_reg[i][id])

    }




}


function showImages(recieved_image){

    modal = document.getElementById("myModal")
    modal.style.display = "block";
    document.getElementById("img01").src = recieved_image.src
    var captionText = document.getElementById("caption");
    captionText.innerHTML = "Accuracy";

    // Get the <span> element that closes the modal
    var span = document.getElementsByClassName("close")[0];
    span.textContent= "X"
    // When the user clicks on <span> (x), close the modal
    span.onclick = function() {
        modal.style.display = "none";
    }
}

function showAcc(){
    showImages(acc_img)
}

function showLoss(){
    showImages(loss_img)
}

function showRecons(){
    showImages(recons_img)
}


function showCode(){
    code_area = document.getElementById("code")
        code_area.innerHTML =code_text
    code_area.rows = 50
    code_area.cols = 150

}
