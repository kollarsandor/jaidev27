import spinal.core._
import spinal.lib._

class TGNUpdateUnit extends Component {
  val io = new Bundle {
    val node_features = in Vec(SInt(16 bits), 2048)
    val memory = in Vec(SInt(16 bits), 2048)
    val time_factor = in SInt(16 bits)
    val updated_memory = out Vec(SInt(16 bits), 2048)
  }

  val pipeline = new Area {
    val stage0 = Reg(Vec(SInt(16 bits), 2048))
    stage0 := io.memory * (1 - io.time_factor) + io.node_features * io.time_factor
    io.updated_memory := stage0
  }
}

object TGNUpdateUnitVerilog {
  def main(args: Array[String]): Unit = {
    SpinalConfig(targetDirectory = "build").generateVerilog(new TGNUpdateUnit)
  }
}
