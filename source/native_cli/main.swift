import Foundation
import SQLite3
import CryptoKit

let external: UInt64 = 0x10000000
let hFlag: UInt64 = 0x01000000
let i8x: UInt64 = 0x8a1e9b
let i4x: UInt64 = 0x8a1eab
let palettes: Set<UInt64> = [0x8a1e4b,0x8a1e5b,0x8a1e6b,0x8a1e7b,0x8a1e8b]
struct QuantizationFormat {
  let name:String;let packed:Int32;let identifier:UInt64;let suffix:String
}
// Packed values and checkpoint identifiers follow the pinned ccv and s4nnc versions.
let quantizationFormats = [
  QuantizationFormat(name:"q4k",packed:1,identifier:0x8a1eab,suffix:"i4x"),
  QuantizationFormat(name:"q3k",packed:2,identifier:0x8a1eac,suffix:"i3x"),
  QuantizationFormat(name:"q2k",packed:3,identifier:0x8a1ead,suffix:"i2x"),
  QuantizationFormat(name:"iq2s",packed:4,identifier:0x8a1eae,suffix:"iq2s"),
  QuantizationFormat(name:"iq2xs",packed:5,identifier:0x8a1eaf,suffix:"iq2xs"),
  QuantizationFormat(name:"iq3s",packed:6,identifier:0x8a1eb0,suffix:"iq3s"),
  QuantizationFormat(name:"iq3xxs",packed:7,identifier:0x8a1eb1,suffix:"iq3xxs"),
  QuantizationFormat(name:"iq2xxs",packed:8,identifier:0x8a1eb2,suffix:"iq2xxs"),
  QuantizationFormat(name:"q5k",packed:9,identifier:0x8a1eb3,suffix:"i5x"),
  QuantizationFormat(name:"q6k",packed:10,identifier:0x8a1eb4,suffix:"i6x"),
]
func targetFormat(_ value:String) throws -> QuantizationFormat {
  let name=value.lowercased().replacingOccurrences(of:"_",with:"")
  guard let format=quantizationFormats.first(where:{$0.name==name || $0.suffix==name}) else {
    throw Failure.message("Unknown format: \(value). Choose \(quantizationFormats.map(\.suffix).joined(separator:", "))")
  }
  return format
}
let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
enum Failure: Error, CustomStringConvertible {
  case message(String)
  var description: String { if case .message(let text) = self { return text }; return "Error" }
}
func require(_ condition: Bool, _ message: String) throws {
  if !condition { throw Failure.message(message) }
}
func integer(_ data: Data, _ offset: Int, _ bytes: Int, big: Bool = false) throws -> UInt64 {
  try require(offset >= 0 && offset+bytes <= data.count, "Truncated tensor metadata")
  var result: UInt64 = 0
  for i in 0..<bytes { let shift = big ? (bytes-1-i)*8 : i*8; result |= UInt64(data[offset+i]) << shift }
  return result
}
func encoded(_ value: UInt64, _ bytes: Int = 8, big: Bool = false) -> Data {
  Data((0..<bytes).map { UInt8(truncatingIfNeeded: value >> ((big ? bytes-1-$0 : $0)*8)) })
}
func base(_ id: UInt64) -> UInt64 { id & ~(external | hFlag) }
func codecName(_ id: UInt64) -> String {
  if let format=quantizationFormats.first(where:{$0.identifier==base(id)}) {return format.suffix}
  return [UInt64(0):"raw",0x511:"ezm7",i8x:"i8x",0x8a1e4b:"q4p",0x8a1e5b:"q5p",0x8a1e6b:"q6p",0x8a1e7b:"q7p",0x8a1e8b:"q8p"][base(id)] ?? ("0x"+String(id,radix:16))
}
func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format:"%02x",$0) }.joined() }
func fileDigest(_ url: URL, progress: ((Int,Int)->Void)? = nil) throws -> String {
  let file=try FileHandle(forReadingFrom:url); defer {try? file.close()}
  let total=Int(try file.seekToEnd());try file.seek(toOffset:0)
  var completed=0
  var hash=SHA256()
  while let chunk=try file.read(upToCount:8<<20),!chunk.isEmpty {hash.update(data:chunk);completed+=chunk.count;progress?(completed,total)}
  return hash.finalize().map {String(format:"%02x",$0)}.joined()
}
func json(_ object: Any) throws -> Data { try JSONSerialization.data(withJSONObject:object,options:[.prettyPrinted,.sortedKeys]) }
func dimensions(_ data: Data) throws -> [Int] {
  try require(data.count % 4 == 0,"Invalid dimension metadata")
  var result=[Int]()
  for i in stride(from:0,to:data.count,by:4) {
    let d=try integer(data,i,4)
    // ccv ends the logical shape at the first zero; remaining slots can hold codec metadata.
    if d==0 {break}
    try require(d<=UInt64(Int32.max),"Invalid tensor dimensions")
    result.append(Int(d))
  }
  return result
}
func count(_ shape:[Int]) throws -> Int {
  var n=1
  for d in shape { let (next,overflow)=n.multipliedReportingOverflow(by:d);try require(!overflow,"Tensor size overflow");n=next }
  return n
}
func itemSize(_ dtype:Int32) throws -> Int {
  switch dtype {case 0x20000,0x80000:return 2;case 0x4000:return 4;case 0x10000:return 8;default:throw Failure.message("Unsupported floating datatype \(dtype)")}
}

final class Database {
  var handle: OpaquePointer?
  init(_ url:URL,readOnly:Bool) throws {
    let flags=readOnly ? SQLITE_OPEN_READONLY : SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE
    if sqlite3_open_v2(url.path,&handle,flags,nil) != SQLITE_OK {throw Failure.message("Cannot open SQLite checkpoint: \(url.path)")}
    sqlite3_busy_timeout(handle,30000)
    try exec("PRAGMA trusted_schema=OFF")
  }
  deinit {sqlite3_close(handle)}
  func exec(_ sql:String) throws {
    if sqlite3_exec(handle,sql,nil,nil,nil) != SQLITE_OK {throw Failure.message(String(cString:sqlite3_errmsg(handle)))}
  }
  func statement(_ sql:String) throws -> OpaquePointer {
    var p:OpaquePointer?
    try require(sqlite3_prepare_v2(handle,sql,-1,&p,nil)==SQLITE_OK,"SQLite: \(String(cString:sqlite3_errmsg(handle)))")
    return p!
  }
  func blob(_ p:OpaquePointer,_ col:Int32) -> Data {
    let n=Int(sqlite3_column_bytes(p,col));guard n>0,let bytes=sqlite3_column_blob(p,col) else{return Data()}
    return Data(bytes:bytes,count:n)
  }
  func integrity() throws {
    let p=try statement("PRAGMA integrity_check");defer{sqlite3_finalize(p)}
    try require(sqlite3_step(p)==SQLITE_ROW && String(cString:sqlite3_column_text(p,0))=="ok","SQLite integrity check failed")
  }
}
struct Tensor {
  let name:String;let type:UInt64;let format:UInt64;let datatype:UInt64;let dim:Data;let length:Int
  var id:UInt64 {type>>32}
  var dtype:Int32 {Int32(truncatingIfNeeded:datatype)}
}
final class Store {
  let url:URL;let db:Database;let file:FileHandle;let trailer:UInt64
  var sidecar:FileHandle?
  var records=[Tensor]()
  init(_ url:URL) throws {
    self.url=url
    try require(!url.path.hasSuffix(".partial") && !url.path.hasSuffix(".part"),"Input download is incomplete")
    file=try FileHandle(forReadingFrom:url)
    let header=try file.read(upToCount:100) ?? Data()
    try require(header.count==100 && header.prefix(16)==Data("SQLite format 3\0".utf8),"Input must be a native Draw Things SQLite checkpoint")
    trailer=try integer(header,60,4,big:true)
    let size=try file.seekToEnd()
    try require(trailer==0 || trailer>=100 && trailer<=size,"Invalid trailer-store offset")
    if trailer==0 && FileManager.default.fileExists(atPath:url.path+"-tensordata") {sidecar=try FileHandle(forReadingFrom:URL(fileURLWithPath:url.path+"-tensordata"))}
    db=try Database(url,readOnly:true)
    let p=try db.statement("SELECT name,type,format,datatype,dim,length(data) FROM tensors ORDER BY name");defer{sqlite3_finalize(p)}
    while sqlite3_step(p)==SQLITE_ROW {
      records.append(Tensor(name:String(cString:sqlite3_column_text(p,0)),type:UInt64(bitPattern:sqlite3_column_int64(p,1)),format:UInt64(bitPattern:sqlite3_column_int64(p,2)),datatype:UInt64(bitPattern:sqlite3_column_int64(p,3)),dim:db.blob(p,4),length:Int(sqlite3_column_int64(p,5))))
    }
    try require(!records.isEmpty,"No tensors found")
  }
  deinit {try? file.close();try? sidecar?.close()}
  func data(_ tensor:Tensor) throws -> Data {
    let p=try db.statement("SELECT data FROM tensors WHERE name=?");defer{sqlite3_finalize(p)}
    sqlite3_bind_text(p,1,tensor.name,-1,transient)
    try require(sqlite3_step(p)==SQLITE_ROW,"Missing tensor: \(tensor.name)")
    var data=db.blob(p,0)
    if tensor.format & (1<<32) != 0 {
      let parts=try db.statement("SELECT part,data FROM tensor_splits WHERE name=? ORDER BY part");defer{sqlite3_finalize(parts)}
      sqlite3_bind_text(parts,1,tensor.name,-1,transient);var expected=0
      while sqlite3_step(parts)==SQLITE_ROW {try require(Int(sqlite3_column_int(parts,0))==expected,"Missing tensor split");data.append(db.blob(parts,1));expected+=1}
    }
    return data
  }
  func payload(_ tensor:Tensor) throws -> (Data,Data) {
    let metadata=try data(tensor)
    if tensor.id & external == 0 {return (metadata,Data())}
    let prefix=palettes.contains(base(tensor.id)) ? Data(metadata.prefix(8)) : Data()
    let offset=try integer(metadata,prefix.count,8),length=try integer(metadata,prefix.count+8,8)
    guard let source=trailer>0 ? file : sidecar else {throw Failure.message("Missing -tensordata sidecar")}
    let (position,overflow)=offset.addingReportingOverflow(trailer)
    let size=try source.seekToEnd()
    try require(!overflow && position<=size && length<=size-position && length<=UInt64(Int.max),"Truncated external tensor: \(tensor.name)")
    try source.seek(toOffset:position)
    let payload=try source.read(upToCount:Int(length)) ?? Data()
    try require(payload.count==Int(length),"Truncated tensor read")
    return (payload,prefix)
  }
}

struct Options {
  var input:URL;var outputDirectory:URL?;var hadamard=true;var inspect=false;var quiet=false
  var format=quantizationFormats[0]
}
final class Progress {
  let quiet:Bool
  let started=Date()
  var lastUpdate=Date.distantPast
  init(_ quiet:Bool) {self.quiet=quiet}
  func update(_ phase:String,_ completed:Int=0,_ total:Int=0,_ detail:String="",force:Bool=false) {
    let now=Date()
    guard !quiet && (force || now.timeIntervalSince(lastUpdate)>=1) else {return}
    lastUpdate=now
    let percent=total>0 ? String(format:" %5.1f%%",100*Double(completed)/Double(total)) : ""
    let elapsed=Int(now.timeIntervalSince(started))
    FileHandle.standardError.write(Data("\(phase)\(percent) | elapsed \(elapsed)s\(detail.isEmpty ? "" : " | "+detail)\n".utf8))
  }
}
func textModel(_ name:String)->Bool {
  name.range(of:"^__text_model(?:_?[0-9]+)?__",options:.regularExpression) != nil
}
func eligible(_ shape:[Int])->Bool {
  shape.count==2 && shape.min()!>1
}
func rotated(_ name:String,_ width:Int,_ enabled:Bool)->Bool {
  // Keep the established text-projection allowlist; dimensions alone do not identify an operator.
  enabled && textModel(name) && width%256==0 && name.range(of:"(?:t-(?:q_proj|k_proj|v_proj|out_proj)-|mlp_(?:gate|up|down)_proj-)",options:.regularExpression) != nil
}
func inspect(_ source:Store,_ hadamard:Bool=true) throws -> [String:Any] {
  var n=0,total=0,rotations=0,codecs=Set<String>(),unsupported=[[String:Any]]()
  for tensor in source.records {
    let shape=try dimensions(tensor.dim)
    if !eligible(shape) {continue}
    n+=1;total+=try count(shape);codecs.insert(codecName(tensor.id))
    if rotated(tensor.name,shape.last!,hadamard) {rotations+=1}
    if !([UInt64(0),0x511,i8x].contains(base(tensor.id)) || palettes.contains(base(tensor.id)) || quantizationFormats.contains(where:{$0.identifier==base(tensor.id)})) || ![Int32(0x20000),0x4000,0x10000,0x80000].contains(tensor.dtype) || tensor.datatype>>32 != 0 {
      unsupported.append(["name":tensor.name,"codec":codecName(tensor.id),"datatype":tensor.datatype])
    }
  }
  return ["input":source.url.path,"total_tensors":source.records.count,"selected_matrices":n,"selected_parameters":total,"source_codecs":codecs.sorted(),"unsupported":unsupported,"hadamard_policy":hadamard ? "auto" : "disabled","hadamard_matrices":rotations,"plain_matrices":n-rotations]
}
func decoded(_ payload:Data,_ tensor:Tensor,_ shape:[Int]) throws -> Data {
  let n=try count(shape),size=try itemSize(tensor.dtype)
  let (bytes,overflow)=n.multipliedReportingOverflow(by:size)
  try require(!overflow,"Tensor byte size overflow")
  switch base(tensor.id) {
  case 0:try require(payload.count==bytes,"Incorrect raw payload size: \(tensor.name)");return payload
  case i8x:
    try require(payload.count==((n+127)&(-128))+n/shape.last!*size && tensor.id&hFlag==0,"Incorrect i8x payload size")
    var result=Data(count:bytes)
    result.withUnsafeMutableBytes {out in payload.withUnsafeBytes {input in ccv_nnc_dequantize_8i_rowwise(input.baseAddress,tensor.dtype,1,payload.count,shape.last!,out.baseAddress,n)}}
    return result
  case 0x511:
    try require(tensor.dtype==0x20000,"EZM7 requires FP16")
    var result=Data(count:bytes)
    let ok=result.withUnsafeMutableBytes {out in payload.withUnsafeBytes {input in dt_decode_ezm7(input.bindMemory(to:UInt8.self).baseAddress,payload.count,out.bindMemory(to:UInt16.self).baseAddress,n)}}
    try require(ok==1,"Invalid EZM7 payload");return result
  default:
    if let format=quantizationFormats.first(where:{$0.identifier==base(tensor.id)}) {
      let packed=format.packed | (tensor.id&hFlag != 0 ? 0x100 : 0)
      let expected=ccv_nnc_8i_rowwise_x_data_size(packed,tensor.dtype,n,shape.last!)
      try require(expected>0 && payload.count==expected,"Incorrect packed rowwise payload size or incompatible H256 width: \(tensor.name)")
      var result=Data(count:bytes)
      result.withUnsafeMutableBytes {out in payload.withUnsafeBytes {input in ccv_nnc_dequantize_8i_rowwise_x(input.baseAddress,tensor.dtype,1,payload.count,shape.last!,packed,out.baseAddress,n)}}
      return result
    }
    try require(palettes.contains(base(tensor.id)),"Unsupported codec: \(codecName(tensor.id))")
    let bits=Int32(((base(tensor.id)>>4)&15))
    var result=Data(count:bytes)
    let ok=result.withUnsafeMutableBytes {out in payload.withUnsafeBytes {input in dt_decode_palette(input.bindMemory(to:UInt8.self).baseAddress,payload.count,tensor.dtype,bits,out.baseAddress,n)}}
    try require(ok==1,"Invalid or unsupported partial palette block: \(tensor.name)");return result
  }
}
func quantize(_ values:Data,_ tensor:Tensor,_ shape:[Int],_ rotation:Bool,_ format:QuantizationFormat) throws -> (Data,Double) {
  let n=try count(shape),width=shape.last!,packed=format.packed | (rotation ? 0x100 : 0)
  let finite=values.withUnsafeBytes {dt_finite($0.baseAddress,tensor.dtype,n)}
  try require(finite==1,"Nonfinite input: \(tensor.name)")
  let size=ccv_nnc_8i_rowwise_x_data_size(packed,tensor.dtype,n,width)
  try require(size>0,"Unsupported native packing shape")
  var output=Data(count:size)
  let actual=output.withUnsafeMutableBytes {out in values.withUnsafeBytes {input in ccv_nnc_quantize_8i_rowwise_x(input.baseAddress,tensor.dtype,1,n,width,packed,nil,0,nil,0,out.baseAddress,size)}}
  try require(actual==size,"Native quantizer failed")
  var check=Data(count:values.count)
  check.withUnsafeMutableBytes {out in output.withUnsafeBytes {input in ccv_nnc_dequantize_8i_rowwise_x(input.baseAddress,tensor.dtype,1,output.count,width,packed,out.baseAddress,n)}}
  let error=values.withUnsafeBytes {a in check.withUnsafeBytes {b in dt_relative_rmse(a.baseAddress,b.baseAddress,tensor.dtype,shape[0],width)}}
  try require(error.isFinite && error<=0.30,"Quantization check failed: \(tensor.name), relative RMSE \(error)")
  return (output,error)
}
func align(_ file:FileHandle) throws -> UInt64 {
  let offset=try file.offset(),aligned=(offset+16383)/16384*16384
  if aligned>offset {try file.write(contentsOf:Data(count:Int(aligned-offset)))}
  return aligned
}
func copy(_ source:URL,_ output:FileHandle, progress: ((Int,Int)->Void)? = nil) throws {
  let file=try FileHandle(forReadingFrom:source);defer{try? file.close()}
  let total=Int(try file.seekToEnd());try file.seek(toOffset:0)
  var completed=0
  while let data=try file.read(upToCount:8<<20),!data.isEmpty {try output.write(contentsOf:data);completed+=data.count;progress?(completed,total)}
}
func convert(_ source:Store,_ destination:URL,_ options:Options) throws -> [String:Any] {
  let fm=FileManager.default
  try require(!fm.fileExists(atPath:destination.path),"Output already exists: \(destination.path)")
  try require(!fm.fileExists(atPath:destination.deletingPathExtension().appendingPathExtension("validation.json").path),"Validation manifest already exists for: \(destination.path)")
  let info=try inspect(source,options.hadamard)
  let matrixTotal=info["selected_matrices"] as! Int,parameterTotal=info["selected_parameters"] as! Int
  try require(matrixTotal>0,"No eligible 2D matrices found for the selected scope")
  let unsupported=info["unsupported"] as! [[String:Any]]
  try require(unsupported.isEmpty,"Unsupported selected source codec or datatype. Run --inspect for details and --help for supported sources. No output was created.")
  let progress=Progress(options.quiet)
  progress.update("Preparing",0,0,"\(matrixTotal) matrices; format: \(options.format.suffix); H256: \(info["hadamard_matrices"]!) auto-selected; unrotated: \(info["plain_matrices"]!)",force:true)
  progress.update("Checking source SQLite",force:true)
  try source.db.integrity()
  let stage=URL(fileURLWithPath:destination.path+".work",isDirectory:true)
  try require(!fm.fileExists(atPath:stage.path),"Intermediate directory already exists: \(stage.path)")
  try fm.createDirectory(at:stage,withIntermediateDirectories:true)
  let databaseURL=stage.appendingPathComponent("metadata.sqlite"),dataURL=stage.appendingPathComponent("tensordata")
  let db=try Database(databaseURL,readOnly:false)
  try db.exec("CREATE TABLE tensors (name TEXT,type INTEGER,format INTEGER,datatype INTEGER,dim BLOB,data BLOB,PRIMARY KEY(name))")
  try db.exec("BEGIN")
  let statement=try db.statement("INSERT INTO tensors VALUES(?,?,?,?,?,?)");defer{sqlite3_finalize(statement)}
  fm.createFile(atPath:dataURL.path,contents:nil)
  let file=try FileHandle(forWritingTo:dataURL);defer{try? file.close()}
  var entries=[[String:Any]](),hashes=[String:String](),converted=0,weights=0
  let started=Date()
  for tensor in source.records {
    let shape=try dimensions(tensor.dim)
    let selected=eligible(shape)
    var (payload,prefix)=try source.payload(tensor)
    var identifier=tensor.id
    var entry:[String:Any]=["name":tensor.name,"shape":shape,"input_codec":String(identifier,radix:16),"input_payload_bytes":payload.count]
    if selected {
      let rotation=rotated(tensor.name,shape.last!,options.hadamard)
      progress.update("Quantizing",weights,parameterTotal,"[\(converted)/\(matrixTotal) complete] \(options.format.suffix) \(rotation ? "H256" : "unrotated") \(tensor.name)",force:true)
      let sourceBytes=palettes.contains(base(tensor.id)) && tensor.id&external != 0 ? Data(prefix.prefix(4))+payload : payload
      let values=try decoded(sourceBytes,tensor,shape)
      let (packed,error)=try quantize(values,tensor,shape,rotation,options.format)
      payload=packed;prefix=Data();identifier=options.format.identifier|(rotation ? hFlag : 0);converted+=1;weights+=try count(shape)
      entry["relative_rmse"]=error;entry["hadamard"]=rotation
      progress.update("Quantizing",weights,parameterTotal,String(format:"[%d/%d complete] %@ -> %.1f MB; relRMSE %.4f",converted,matrixTotal,tensor.name,Double(payload.count)/1e6,error))
    }
    let hash=digest(payload);hashes[tensor.name]=hash
    var metadata=payload
    if identifier&external != 0 || selected || identifier==0 && payload.count>=65536 {
      let offset=try align(file);try file.write(contentsOf:payload)
      metadata=prefix+encoded(offset)+encoded(UInt64(payload.count));identifier|=external
    }
    let type=(identifier<<32)|(tensor.type&0xffffffff)
    sqlite3_reset(statement);sqlite3_clear_bindings(statement)
    sqlite3_bind_text(statement,1,tensor.name,-1,transient)
    sqlite3_bind_int64(statement,2,Int64(bitPattern:type));sqlite3_bind_int64(statement,3,Int64(tensor.format&0xffffffff))
    sqlite3_bind_int64(statement,4,Int64(bitPattern:tensor.datatype))
    _=tensor.dim.withUnsafeBytes {sqlite3_bind_blob(statement,5,$0.baseAddress,Int32(tensor.dim.count),transient)}
    _=metadata.withUnsafeBytes {sqlite3_bind_blob(statement,6,$0.baseAddress,Int32(metadata.count),transient)}
    try require(sqlite3_step(statement)==SQLITE_DONE,"SQLite tensor write failed: \(tensor.name)")
    entry["output_codec"]=String(identifier,radix:16);entry["output_payload_bytes"]=payload.count;entry["payload_sha256"]=hash;entries.append(entry)
  }
  progress.update("Quantizing",weights,parameterTotal,"[\(converted)/\(matrixTotal) complete]",force:true)
  try file.synchronize();try file.close();try db.exec("COMMIT");try db.integrity()
  let assembled=stage.appendingPathComponent("assembled.ckpt")
  fm.createFile(atPath:assembled.path,contents:nil)
  let output=try FileHandle(forWritingTo:assembled)
  progress.update("Assembling checkpoint",force:true)
  try copy(databaseURL,output);let trailer=try align(output)
  try require(trailer<=UInt64(UInt32.max),"Metadata exceeds native trailer offset limit")
  try copy(dataURL,output,progress:{progress.update("Assembling checkpoint",$0,$1)});try output.seek(toOffset:60);try output.write(contentsOf:encoded(trailer,4,big:true));try output.synchronize();try output.close()
  progress.update("Checking output SQLite",force:true)
  let final=try Store(assembled);try final.db.integrity()
  try require(final.records.map(\.name)==source.records.map(\.name),"Final tensor names differ")
  progress.update("Verifying tensors",0,final.records.count,force:true)
  for (index,tensor) in final.records.enumerated() {let(payload,_)=try final.payload(tensor);try require(digest(payload)==hashes[tensor.name],"Final payload hash mismatch: \(tensor.name)");progress.update("Verifying tensors",index+1,final.records.count)}
  let outputSize=(try fm.attributesOfItem(atPath:assembled.path)[.size] as! NSNumber).uint64Value
  progress.update("Hashing checkpoint",force:true)
  let hash=try fileDigest(assembled,progress:{progress.update("Hashing checkpoint",$0,$1)})
  let manifest:[String:Any]=["input":source.url.path,"output_format":options.format.name,"output_bytes":outputSize,"converted_tensors":converted,"converted_elements":weights,"total_tensors":entries.count,"sha256":hash,"elapsed_seconds":Date().timeIntervalSince(started),"hadamard_policy":info["hadamard_policy"]!,"hadamard_matrices":info["hadamard_matrices"]!,"plain_matrices":info["plain_matrices"]!,"tensors":entries,"upstream_ccv":"810fb2992bd4b5463052a8f94338c8bffc273cc6"]
  try fm.linkItem(at:assembled,to:destination)
  try json(manifest).write(to:destination.deletingPathExtension().appendingPathExtension("validation.json"),options:.atomic)
  // Only this invocation's generated work directory is removed; input and sidecars are untouched.
  try fm.removeItem(at:stage)
  progress.update("Finished",1,1,force:true)
  return manifest
}

func outputURL(_ source:URL,_ folder:URL,_ format:QuantizationFormat)->URL {
  let stem=source.deletingPathExtension().lastPathComponent.replacingOccurrences(of:"_(?:i[2-8]x|iq[23](?:s|xs|xxs)|q[2-6]k|q[4-8]p|f16|fp16|bf16|f32|fp32)$",with:"",options:[.regularExpression,.caseInsensitive])
  return folder.appendingPathComponent(stem+"_"+format.suffix+".ckpt")
}
func importerExecutable(_ override:String?) throws -> URL {
  let candidates:[String]
  if let override {candidates=[(override as NSString).expandingTildeInPath]} else {
    candidates=(ProcessInfo.processInfo.environment["PATH"] ?? "").components(separatedBy:":").map {URL(fileURLWithPath:$0.isEmpty ? FileManager.default.currentDirectoryPath : $0).appendingPathComponent("draw-things-cli").path}
  }
  guard let path=candidates.first(where:{FileManager.default.isExecutableFile(atPath:$0)}) else {
    throw Failure.message("Safetensors import requires draw-things-cli on PATH or --draw-things-cli PATH. Native .ckpt conversion needs no importer.")
  }
  return URL(fileURLWithPath:path).standardizedFileURL
}
func runImport(_ source:URL,_ stage:URL,_ executable:URL,_ quiet:Bool,_ dryRun:Bool=false) throws -> URL {
  let fm=FileManager.default,models=stage.appendingPathComponent("models",isDirectory:true)
  try require(fm.fileExists(atPath:source.path),"Input does not exist: \(source.path)")
  try fm.createDirectory(at:models,withIntermediateDirectories:false)
  let process=Process(),pipe=Pipe(),logURL=stage.appendingPathComponent("import.log")
  fm.createFile(atPath:logURL.path,contents:nil)
  let log=try FileHandle(forWritingTo:logURL);defer{try? log.close()}
  process.executableURL=executable
  process.arguments=["models","import",source.path,"--models-dir",models.path,"--offline","--no-download-missing"]+(dryRun ? ["--dry-run"] : [])
  process.standardInput=FileHandle.nullDevice;process.standardOutput=pipe;process.standardError=pipe
  try process.run()
  try pipe.fileHandleForWriting.close()
  while let data=try pipe.fileHandleForReading.read(upToCount:4096),!data.isEmpty {
    try log.write(contentsOf:data)
    if !quiet {try (dryRun ? FileHandle.standardOutput : FileHandle.standardError).write(contentsOf:data)}
  }
  try pipe.fileHandleForReading.close();process.waitUntilExit()
  try require(process.terminationStatus==0,"Official Draw Things importer exited with status \(process.terminationStatus). Details: \(logURL.path)")
  return models
}
func inspectSafetensors(_ source:URL,_ executable:URL) throws {
  let fm=FileManager.default,stage=fm.temporaryDirectory.appendingPathComponent("dt-quant-inspect-"+UUID().uuidString,isDirectory:true)
  try fm.createDirectory(at:stage,withIntermediateDirectories:false);defer{try? fm.removeItem(at:stage)}
  _=try runImport(source,stage,executable,false,true)
}
func convertSafetensors(_ source:URL,_ destination:URL,_ options:Options,_ executable:URL) throws -> [String:Any] {
  let fm=FileManager.default,specURL=destination.deletingPathExtension().appendingPathExtension("model.json")
  for path in [destination,specURL,destination.deletingPathExtension().appendingPathExtension("validation.json")] {
    try require(!fm.fileExists(atPath:path.path),"Output already exists: \(path.path)")
  }
  let stage=URL(fileURLWithPath:destination.path+".import.work",isDirectory:true)
  try require(!fm.fileExists(atPath:stage.path),"Import work directory already exists: \(stage.path)")
  try fm.createDirectory(at:stage,withIntermediateDirectories:true)
  let models=try runImport(source,stage,executable,options.quiet)
  let customURL=models.appendingPathComponent("custom.json")
  let object=try JSONSerialization.jsonObject(with:Data(contentsOf:customURL))
  guard let specifications=object as? [[String:Any]],specifications.count==1,var specification=specifications.first,
        let name=specification["file"] as? String,(name as NSString).lastPathComponent==name else {
    throw Failure.message("Importer produced no unique primary model specification. See \(stage.appendingPathComponent("import.log").path)")
  }
  let imported=models.appendingPathComponent(name)
  try require(fm.fileExists(atPath:imported.path),"Importer produced no primary checkpoint: \(imported.path)")
  let companions=try fm.contentsOfDirectory(at:models,includingPropertiesForKeys:nil).filter {
    $0.lastPathComponent != name && $0.lastPathComponent != name+"-tensordata" && ($0.pathExtension=="ckpt" || $0.lastPathComponent.hasSuffix(".ckpt-tensordata"))
  }.sorted {$0.lastPathComponent<$1.lastPathComponent}
  for file in companions {
    let target=destination.deletingLastPathComponent().appendingPathComponent(file.lastPathComponent)
    try require(!fm.fileExists(atPath:target.path),"Imported companion output already exists: \(target.path)")
  }
  var manifest=try convert(Store(imported),destination,options)
  for file in companions {try fm.linkItem(at:file,to:destination.deletingLastPathComponent().appendingPathComponent(file.lastPathComponent))}
  specification["file"]=destination.lastPathComponent
  try json(specification).write(to:specURL,options:.withoutOverwriting)
  manifest["input"]=source.path;manifest["importer"]=executable.path;manifest["imported_model_version"]=specification["version"]
  manifest["model_specification"]=specURL.lastPathComponent;manifest["imported_companions"]=companions.map(\.lastPathComponent)
  try json(manifest).write(to:destination.deletingPathExtension().appendingPathExtension("validation.json"),options:.atomic)
  try fm.removeItem(at:stage)
  return manifest
}

let help="""
dt-i4x 0.2.1
Quantize a Draw Things checkpoint or safetensors model.

Usage: dt-i4x INPUT [-o DIR] [OPTIONS]

Options:
  -o, --output-dir DIR    Output folder (default: input folder)
  --format FORMAT        Output quantization (default: i4x)
  --inspect              Inspect the input without converting
  --no-hadamard          Disable automatic H256 rotation
  --draw-things-cli PATH  Safetensors importer (default: found on PATH)
  -q, --quiet            Print only the finished output path
  --version              Print version
  -h, --help             Show this help

Formats:
  Standard:  i2x, i3x, i4x, i5x, i6x
  IQ:        iq2xxs, iq2xs, iq2s, iq3xxs, iq3s

Defaults:
  The input defines what gets quantized. Convolutions and vectors are preserved.
  H256 is automatic for supported operators. Progress appears during conversion.

Safetensors:
  Requires the official draw-things-cli importer; import runs offline.

Examples:
  dt-i4x model.ckpt --format i3x
  dt-i4x model.safetensors -o converted
"""
func run() throws {
  let args=Array(CommandLine.arguments.dropFirst())
  if args.isEmpty || args.contains("--help") || args.contains("-h") {print(help);return}
  if args==["--version"] {print("0.2.1");return}
  var input:String?,directory:String?,importer:String?,hadamard=true,inspection=false,quiet=false,index=0
  var format=quantizationFormats[0]
  while index<args.count {
    let arg=args[index]
    switch arg {
    case "-o","--output-dir":index+=1;try require(index<args.count,"Missing output folder");directory=args[index]
    case "--format":index+=1;try require(index<args.count,"Missing output format");format=try targetFormat(args[index])
    case "--draw-things-cli":index+=1;try require(index<args.count,"Missing importer path");importer=args[index]
    case "--inspect":inspection=true
    case "--no-hadamard":hadamard=false
    case "--quiet","-q":quiet=true
    default:try require(!arg.hasPrefix("-") && input==nil,"Unexpected argument: \(arg)");input=arg
    }
    index+=1
  }
  guard let input else {throw Failure.message("Missing input checkpoint")}
  let sourceURL=URL(fileURLWithPath:(input as NSString).expandingTildeInPath).standardizedFileURL.resolvingSymlinksInPath()
  let folder=directory.map {URL(fileURLWithPath:($0 as NSString).expandingTildeInPath,isDirectory:true).standardizedFileURL} ?? sourceURL.deletingLastPathComponent()
  let options=Options(input:sourceURL,outputDirectory:folder,hadamard:hadamard,inspect:inspection,quiet:quiet,format:format)
  let destination=outputURL(sourceURL,folder,format)
  try require(inspection || destination.standardizedFileURL != sourceURL,"Input already has the target filename; choose another output folder")
  let manifest:[String:Any]
  if sourceURL.pathExtension.lowercased()=="safetensors" {
    let executable=try importerExecutable(importer)
    if inspection {try inspectSafetensors(sourceURL,executable);return}
    manifest=try convertSafetensors(sourceURL,destination,options,executable)
  } else {
    let source=try Store(sourceURL)
    if inspection {print(String(data:try json(inspect(source,hadamard)),encoding:.utf8)!);return}
    manifest=try convert(source,destination,options)
  }
  print(destination.path)
  if !quiet {print(String(format:"%.2f GiB; %d matrices converted; all %d tensor payloads verified.",Double((manifest["output_bytes"] as! NSNumber).uint64Value)/pow(1024,3),manifest["converted_tensors"] as! Int,manifest["total_tensors"] as! Int))}
}
do {try run()} catch {FileHandle.standardError.write(Data("dt-i4x: \(error)\n".utf8));exit(1)}
